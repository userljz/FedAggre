cd /home/ljz/FedAggre

#must config
device=6
yaml=ours.yaml



CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                --dirichlet_alpha 0.01 \
                                                --use_before_fc_emb 0 \
                                                --local_lr 0.001 \
                                                --logfile_info "1227-cifar100_20Clients_Alpha0.01_LR0.001_NoBefore" \
                                                --client_dataset "cifar100" \
                                                --wandb_project "20231224"


