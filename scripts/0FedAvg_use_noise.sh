cd /home/ljz/FedAggre

#must config
device=4
yaml=FedAvg.yaml

filename="0322-cifar100-FedAvgWithNoise-Alpha"


for iter in 0.01 0.05 0.1 1
do
    CUDA_VISIBLE_DEVICES=$device python run_baseline.py --yaml_name $yaml \
                                                        --dirichlet_alpha ${iter} \
                                                        --use_before_fc_emb 1 \
                                                        --local_lr 0.01 \
                                                        --logfile_info ${filename}${iter} \
                                                        --client_dataset "cifar100" \
                                                        --wandb_project "20240322ICMLRebuttal" \
                                                        --round 40 \
                                                        --select_client_num 15
done
