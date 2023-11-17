from typing import Callable, Union, Optional, List, Tuple
from addict import Dict
import torch
import time

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from utils.fl_utils import FlowerClient, test, get_parameters, set_parameters, AverageMeter, \
    SoftTarget, accuracy
from utils.distill_utils import train, distill_test, adjust_lr, ensemble_test
from utils.log_utils import cus_logger
import flwr as fl
from network import define_tsnet
import os
from logging import info
import wandb


class FedCustom(fl.server.strategy.Strategy):
    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            args=None,
            evaluate_fn=None
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.args = args
        self.evaluate_fn = evaluate_fn
        wandb.init(project=args.cfg["wandb_project"], name = args.cfg["logfile_info"], config= args.cfg)
        self.logger_cus = cus_logger(self.args, __name__)
        self.logger_cus.info("\n---Start Using Custom Strategy")

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters.
           No need to use initialize parameter """
        self.logger_cus.info("No need to use initialize parameter")
        return None

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.logger_cus.info('\n---Configure Fit')
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        # n_clients = len(clients)
        # half_clients = n_clients // 2
        standard_config = {"lr": 0.01}
        # higher_lr_config = {"lr": 0.01}
        fit_configurations = []
        # for idx, client in enumerate(clients):
        #     if idx < half_clients:
        #         fit_configurations.append((client, FitIns(parameters, standard_config)))
        #     else:
        #         fit_configurations.append(
        #             (client, FitIns(parameters, higher_lr_config))
        #         )
        for idx, client in enumerate(clients):
            fit_configurations.append(
                (client, FitIns(parameters, standard_config))
            )
        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        self.logger_cus.info('\n---Enter Aggregate Fit')

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        # [(client1's parameters, client1's num_examples), (client2's ...), ...]

        cfg = self.args.cfg
        # Use the existing trained server model
        if self.args.trained_server_model:
            server_net = self.args.trained_server_model
        else:
            server_net = define_tsnet(self.args, name=cfg.server_model, num_class=cfg.num_class)

        client_net = define_tsnet(self.args, name=cfg.client_model, num_class=cfg.num_class)
        client_net_list = []
        self.client_train_num = []
        for index in range(cfg.num_clients):
            (param, data_num_per_client) = weights_results[index]
            # self.logger_cus.info(f'{data_num_per_client = }')
            _client_net = set_parameters(client_net, param)
            _client_net.eval()
            for param in _client_net.parameters():
                param.requires_grad = False
            client_net_list.append(_client_net)
            self.client_train_num.append(data_num_per_client)

        self.logger_cus.info(f'{self.client_train_num = }')

        # ensemble evaluate accuracy before distillation
        _, accu = ensemble_test(self.args, client_net_list, self.client_train_num, self.args.testloader)
        self.logger_cus.info(f'Round {server_round} ensemble evaluate accuracy: {accu}')
        wandb.log({f"Ensemble evaluate accuracy": accu})

        # Fedavg evaluate accuracy before distillation
        parameters_fedavg = ndarrays_to_parameters(aggregate(weights_results))
        _, accu = self.my_evaluate(server_round, parameters_fedavg)
        self.logger_cus.info(f'Round {server_round} Fedavg evaluate accuracy: {accu}')
        wandb.log({f"Fedavg evaluate accuracy": accu})

        self.logger_cus.info(f'\n---Round[{server_round}/{self.args.cfg.round}] Start Distillation-----------')
        # 'snet': student net; 'tnet': teacher net
        server_distill_model_list = {'snet': server_net, 'tnet': client_net_list}
        self.logger_cus.info('Start distillation from fitted client nets to advanced server net')
        self.args.distill_info = 'Client2Server'
        distilled_model = self.model_distillation(server_distill_model_list, self.client_train_num)
        self.args.trained_server_model = distilled_model

        self.logger_cus.info('Start distillation from advanced server net to client net')
        self.args.distill_info = 'Server2Client'
        client_distill_model_list = {'snet': client_net, 'tnet': [distilled_model]}
        client_model = self.model_distillation(client_distill_model_list, [1])

        param_distilled = ndarrays_to_parameters(get_parameters(client_model))

        self.logger_cus.info('-----------End distillation----------------')

        metrics_aggregated = {}
        return param_distilled, metrics_aggregated

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def my_evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        parameters = parameters_to_ndarrays(parameters)
        testloader = self.args.testloader
        set_parameters(self.args.net, parameters)  # Update model with the latest parameters
        loss, accuracy = test(self.args, self.args.net, testloader)
        return loss, accuracy
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def model_distillation(self, model_list, data_num):
        """
        Fitted_client_parameters (e.g.RN8) -> an advanced network (e.g.RN50)

        Args:
            model_list({'snet': server_net, 'tnet': client_net_list}): the fitted client parameters

        Returns:
            nets: the student net(i.e. the distilled server net)
        """
        cfg = self.args.cfg
        distill_info = self.args.distill_info


        # define loss functions
        criterionST = SoftTarget(4)
        optimizer = torch.optim.SGD(model_list['snet'].parameters(),
                                    lr=cfg.distill_lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)
        train_loader, test_loader = self.args.trainloader_d, self.args.testloader_d

        # warp nets and criterions for train and test
        criterions = {'criterionST': criterionST}

        previous_accu = None
        best_model = None
        for epoch in range(1, cfg.distill_epoch):
            adjust_lr(optimizer, epoch)

            # train one epoch
            epoch_start_time = time.time()
            model_list = train(self.args, train_loader, model_list, optimizer, criterions, epoch, distill_info, data_num)

            # evaluate on testing set
            self.logger_cus.info('Testing the models......')
            test_top1, test_top5 = distill_test(test_loader, model_list)
            self.logger_cus.info(f'Prec@1: {test_top1}, Prec@5: {test_top5}')
            wandb.log({f"[{distill_info}]Test_afterDistill": test_top1})

            # save the best
            if previous_accu is None:
                previous_accu = test_top1
                best_model = model_list['snet']
            elif test_top1 > previous_accu:
                best_model = model_list['snet']

            epoch_duration = time.time() - epoch_start_time
            self.logger_cus.info('Epoch time: {}s'.format(int(epoch_duration)))

        return best_model