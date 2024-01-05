cd /home/ljz/FedAggre

#must config
device=4
yaml=FedAvg.yaml

filename="1228Result-cifar100-FedAvg-Alpha"


for iter in 0.01 0.05 0.1 0.5 1
do
    CUDA_VISIBLE_DEVICES=$device python run_baseline.py --yaml_name $yaml \
                                                        --dirichlet_alpha ${iter} \
                                                        --use_before_fc_emb 0 \
                                                        --local_lr 0.01 \
                                                        --logfile_info ${filename}${iter} \
                                                        --client_dataset "cifar100" \
                                                        --wandb_project "1228Result" \
                                                        --round 40 \
                                                        --select_client_num 15
done
