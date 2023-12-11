from addict import Dict
import torch
import argparse
from collections import defaultdict
from network import ClipModel_from_generated
import wandb
import copy

from Utils.cfg_utils import read_yaml
from Utils.data_utils import load_dataloader_from_generate, load_dataloader
from Utils.log_utils import cus_logger
from Utils.typing_utils import FitRes, FitIns
from Utils.server_client_utils import SupConLoss, train, test, get_parameters, set_parameters, CategoryEmbedding
from Utils.flwr_utils import aggregate





parser = argparse.ArgumentParser()
parser.add_argument('--yaml_name', type=str, default='basic.yaml', help='config file name')
parser.add_argument('--margin', type=float)
parser.add_argument('--use_mlp', type=int)
parser.add_argument('--local_lr', type=float)
parser.add_argument('--logfile_info', type=str)
args = parser.parse_args()
args_command = vars(args)

config_basic = read_yaml('basic.yaml')
cfg_basic = Dict(config_basic)
config_specific = read_yaml(args_command['yaml_name'])
cfg_basic.update(config_specific)
cfg = cfg_basic
args = Dict()
args.cfg = cfg
for k,v in args_command.items():
    if v is not None:
        args.cfg[k] = v

# ------ for test ------
if args.cfg.logfile_info == 'test':
    if args.cfg.use_softmax == 0:
        print('args.cfg.use_softmax == 0')
    print(type(args.cfg.use_softmax))


# ------wandb config------
if args.cfg.wandb:
    wandb.init(project=args.cfg["wandb_project"], name=args.cfg["logfile_info"], config=args.cfg)



# ------log config------
log_name = f'{cfg.wandb_project}_{cfg.logfile_info}_{cfg.client_dataset}_{cfg.round}_{cfg.num_clients}'
args.log_name = log_name
logger1 = cus_logger(args, __name__, m='w')


def print_dict(args):
    dict_info = copy.deepcopy(args)

    def info_dict(dict_in):
        for key, value in dict_in.items():
            if isinstance(value, dict):
                info_dict(value)
            else:
                logger1.info(f'{key} : {value}')
    
    info_dict(dict_info)


# ------start training------
args.device = args.cfg.device
logger1.info(f"-------------------------------------------Start a New------------------------------------------------")
logger1.info(f"Training on {args.cfg.device} using PyTorch {torch.__version__}")

print_dict(args)

# ------ Initialize CLIP text emb ------
text_emb = torch.load(f'/home/ljz/dataset/{args.cfg.client_dataset}_generated/{args.cfg.client_dataset}_RN50_textemb.pth')
if args.cfg.reverse_textemb == 1:
    text_emb = text_emb.flip(dims=[0])

args.text_emb = text_emb.float().to(args.cfg.device)


# ------ Initialize class embedding collector ------
args.catemb = CategoryEmbedding()

# ------ Initialize DataLoader & Server ------
train_loaders, test_loader = load_dataloader_from_generate(args, args.cfg.client_dataset, dataloader_num=args.cfg.num_clients)


def client_fn(cid, _args):
    _model = ClipModel_from_generated(_args)
    _train_loader = train_loaders[cid]
    _test_loader = test_loader
    return Client(_args, cid, _model, _train_loader, _test_loader)


def count_labels(data_loader):
    """
    In a non-iid setting, count the data distribution in each client's dataloader
    :param data_loader: [DataLoader1, DataLoader2, ...]
    :return: a Dict
    """
    label_counts = {}
    for client_indice, loader_i in enumerate(data_loader):
        _label_counts = {}
        _data_loader = loader_i
        for _, labels in _data_loader:
            # If labels are not already on CPU, move them
            labels = labels.cpu()
            for label in labels:
                # If label is tensor, convert to python number
                if isinstance(label, torch.Tensor):
                    label = label.item()
                # Increment the count for this label
                _label_counts[label] = _label_counts.get(label, 0) + 1
        _label_counts = dict(sorted(_label_counts.items()))
        label_counts[f'client{client_indice}'] = _label_counts

    return label_counts


class Server():
    def __init__(self, args):
        self.args = args
        self.BeforeEmb_avg = defaultdict(list, {})
        self.AfterEmb_avg = defaultdict(list, {})

    def aggregate_fit(self, server_round, results):
        weights_results = [
            (fit_res.parameters, fit_res.num_examples)
            for fit_res in results
        ]

        parameters_aggregated = aggregate(weights_results)

        if self.args.cfg.use_before_fc_emb == 1:
            for fit_res in results:
                _client_dict = fit_res.metrics
                _BeforeEmb_avg = _client_dict['BeforeEmb_avg']
                self.BeforeEmb_avg = self.args.catemb.merge_mean_var(self.BeforeEmb_avg, _BeforeEmb_avg)

        if self.args.cfg.use_after_fc_emb == 1:
            for fit_res in results:
                _client_dict = fit_res.metrics
                _AfterEmb_avg = _client_dict['AfterEmb_avg']
                self.AfterEmb_avg = self.args.catemb.merge(self.AfterEmb_avg, _AfterEmb_avg)

        return parameters_aggregated

    def configure_fit(self, server_round, parameters):
        config = {'server_round': server_round, 'BeforeEmb_avg': self.BeforeEmb_avg, 'AfterEmb_avg': self.AfterEmb_avg}

        fit_ins = FitIns(parameters, config)

        return fit_ins

    def server_conduct(self, server_round, results):
        parameters_aggregated = self.aggregate_fit(server_round, results)
        fit_ins = self.configure_fit(server_round, parameters_aggregated)
        return fit_ins


class Client():
    def __init__(self, args, cid, net, train_loader, test_loader):
        self.cid = cid
        self.net = net
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = SupConLoss(self.args, self.args.text_emb)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=float(self.args.cfg.local_lr),
                                    momentum=float(self.args.cfg.local_momentum),
                                    weight_decay=float(self.args.cfg.local_weight_decay))


        # ------ train multiple epochs ------
        before_fc_emb = defaultdict(list, {})
        after_fc_emb = defaultdict(list, {})
        extra_loss_embedding, criterion_extra = None, None
        if config['server_round'] > self.args.cfg.use_after_fc_emb_round and self.args.cfg.use_after_fc_emb == 1:
            extra_loss_embedding = self.args.catemb.generate_train_emb(config['AfterEmb_avg'])
            extra_loss_embedding = extra_loss_embedding.to(self.args.device)
            criterion_extra = SupConLoss(self.args, extra_loss_embedding)
            
        p_images_group, p_labels_group = None, None
        # print('in 0')
        # print(f'{self.args.cfg.use_before_fc_emb=}')
        # print(f'{self.args.cfg.use_before_fc_emb_round=}')
        # print(f"{config['server_round']=}")
        if config['server_round'] > self.args.cfg.use_before_fc_emb_round and self.args.cfg.use_before_fc_emb == 1:
            # print('in 1')
            # p_images_group, p_labels_group = args.catemb.generate_pseudo_data(config['BeforeEmb_avg'], self.args.cfg.gene_num, self.args.cfg.batch_size, self.args.cfg.more_dense)
            p_images_group, p_labels_group = config['p_images_group'], config['p_labels_group']
        

        for epoch in range(1, self.args.cfg.local_epoch + 1):
            client_loss, before_fc_emb, after_fc_emb = train(self.args, self.train_loader, self.net, self.criterion,
                                                             optimizer, epoch, self.cid, config, extra_loss_embedding,
                                                             criterion_extra, before_fc_emb, after_fc_emb, p_images_group, p_labels_group)
            logger1.info(
                f"Train_Client{self.cid}:[{epoch}/{self.args.cfg.local_epoch}], Train_loss:{client_loss:.3f}, DataLength:{len(self.train_loader.dataset)}")
            if self.args.cfg.wandb: wandb.log({f"Train_Client{self.cid}|Train_loss:": client_loss})

        # ------ test accuracy after training ------
        client_loss, accuracy = test(self.args, self.net, self.test_loader, self.criterion)
        logger1.info(f'Val_Client[{self.cid}], Accu[{accuracy:.3f}], Loss[{client_loss:.3f}]')
        if self.args.cfg.wandb: wandb.log({f"Val_Client{self.cid}|Accu:": accuracy})

        # ------ save client model ------
        if self.args.cfg.save_client_model == 1:
            torch.save(self.net, f"./save_client_model/client_{self.cid}.pth")
            logger1.info(f'Finish saving client model {self.cid}')

        # ------ Use Before&After Embedding ------
        BeforeEmb_avg, AfterEmb_avg = 0, 0
        if self.args.cfg.use_before_fc_emb == 1:
            BeforeEmb_avg = self.args.catemb.avg_mean_var(before_fc_emb)
        if self.args.cfg.use_after_fc_emb == 1:
            AfterEmb_avg = self.args.catemb.avg(after_fc_emb)

        return FitRes(get_parameters(self.net), len(self.train_loader),
                      {'BeforeEmb_avg': BeforeEmb_avg, 'AfterEmb_avg': AfterEmb_avg})

    def evaluate(self):
        loss, _accuracy = test(self.args, self.net, self.test_loader, self.criterion)
        logger1.info(f'Cid[{self.cid}], Accu[{float(_accuracy)}]')  # ljz
        return float(loss), float(_accuracy)  # ljz


# -------- Start FL ------
server = Server(args)
label_counts = count_labels(train_loaders)
for key, value in label_counts.items():
    logger1.info(f"[{key}]: {value}")

round_count = 0
while round_count < args.cfg.round:
    round_count += 1
    logger1.info(f'*** Round [{round_count}] ***')
    # ------ Init First Round ------
    if round_count == 1:
        param = get_parameters(ClipModel_from_generated(args))
        fit_ins = FitIns(param, {'server_round': 1})

    # ------ Train Clients ------
    results = []
    for client_id in range(args.cfg.num_clients):
        client = client_fn(client_id, args)

        # --- Fit One Client ---
        fit_res = client.fit(fit_ins.parameters, fit_ins.config)
        results.append(fit_res)

        # --- Evaluate One Client
        # loss, accu = client.evaluate()

    # ------ Server aggregate ------
    fit_ins = server.server_conduct(round_count, results)

    # ------ Test Aggregate Net ------
    with torch.no_grad():
        criterion = SupConLoss(args, args.text_emb)
        aggregate_net = ClipModel_from_generated(args)
        aggregate_net = set_parameters(aggregate_net, fit_ins.parameters)
        loss, accu = test(args, aggregate_net, test_loader, criterion)
    logger1.info(f'*** Round[{round_count}]: Server_Test_Accu[{accu:.3f}] *** ')
    if args.cfg.wandb: wandb.log({f"Server Test Accu": accu, 'epoch': round_count})

    # ------ Generate BeforeFcEmb ------
    if fit_ins.config['server_round'] >= args.cfg.use_before_fc_emb_round and args.cfg.use_before_fc_emb == 1:
        p_images_group, p_labels_group = args.catemb.generate_pseudo_data(fit_ins.config['BeforeEmb_avg'], args.cfg.gene_num, args.cfg.batch_size, args.cfg.more_dense)
        fit_ins.config['p_images_group'], fit_ins.config['p_labels_group'] = p_images_group, p_labels_group


logger1.info(f"-------------------------------------------End All------------------------------------------------")