# 20240323 For Private estimate meancd /home/ljz/FedAggre

#must config
device=7
yaml=train1.yaml

filename="0323-cifar100-PrivateEstimate-Alpha"


for iter in 1 0.1 0.05 0.01
do
    CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                    --dirichlet_alpha ${iter} \
                                                    --use_before_fc_emb 1 \
                                                    --local_lr 0.001 \
                                                    --logfile_info ${filename}${iter} \
                                                    --client_dataset "cifar100" \
                                                    --wandb_project "20240322ICMLRebuttal" \
                                                    --round 40 \
                                                    --select_client_num 15 \
                                                    --private_estimate 1
done