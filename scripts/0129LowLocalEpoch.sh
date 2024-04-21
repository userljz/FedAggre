cd /home/ljz/FedAggre

#must config
device=6
yaml=train1.yaml

filename="0129-cifar100-LocalEpoch10-Alpha"


for iter in 0.01
do
    CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                    --dirichlet_alpha ${iter} \
                                                    --use_before_fc_emb 1 \
                                                    --local_lr 0.005 \
                                                    --logfile_info ${filename}${iter} \
                                                    --client_dataset "cifar100" \
                                                    --wandb_project "0129" \
                                                    --select_client_num 15 \
                                                    --local_epoch 10 \
                                                    --round 400
done