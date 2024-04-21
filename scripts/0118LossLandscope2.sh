cd /home/ljz/FedAggre

#must config
device=6
yaml=train1.yaml

filename="LossLandscape-cifar100-Alpha"


for iter in 0.05
do
    CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                    --dirichlet_alpha ${iter} \
                                                    --use_before_fc_emb 1 \
                                                    --local_lr 0.005 \
                                                    --logfile_info ${filename}${iter} \
                                                    --client_dataset "cifar100" \
                                                    --wandb_project "0116TheoreticalBound" \
                                                    --round 40 \
                                                    --select_client_num 15 \
                                                    --save_model_param 1
done
