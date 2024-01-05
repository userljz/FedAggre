cd /home/ljz/FedAggre

#must config
device=7
yaml=FedProx.yaml


# CUDA_VISIBLE_DEVICES=$device python run_baseline.py --yaml_name $yaml  #&> ./log/test

# filename="1227-EMNIST-FedProx-alpha-"
# wandb_project='20231227EMNIST'

# for dirichlet_alpha in 0.01 0.05 0.1 0.5 1
# do
#   CUDA_VISIBLE_DEVICES=$device python run_baseline.py \
#   --yaml_name $yaml \
#   --dirichlet_alpha $dirichlet_alpha \
#   --logfile_info ${filename}${dirichlet_alpha} \
#   --wandb_project $wandb_project
# done
CUDA_VISIBLE_DEVICES=$device python run_baseline.py --yaml_name $yaml \
                                                --dirichlet_alpha 0.01 \
                                                --use_before_fc_emb 0 \
                                                --local_lr 0.1 \
                                                --logfile_info "1227-FedProx_cifar100_20Clients_Alpha0.01_LR0.1" \
                                                --client_dataset "cifar100" \
                                                --wandb_project "20231224"