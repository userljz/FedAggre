cd /home/ljz/FedAggre

#must config
device=5
yaml=FedProx.yaml


# CUDA_VISIBLE_DEVICES=$device python run_baseline.py --yaml_name $yaml  #&> ./log/test

filename="1227-CIFAR100-FedProx-LR-"
wandb_project='20231227FedProx_GridSearch'

for local_lr in 0.001 0.01 0.1
do
  CUDA_VISIBLE_DEVICES=$device python run_baseline.py --wandb_project $wandb_project --yaml_name $yaml --local_lr $local_lr --logfile_info ${filename}${local_lr} 
done