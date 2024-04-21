cd /home/ljz/FedAggre

#must config
device=7
yaml=FedAvg.yaml

filename="LossLandscape(FedAvg)-cifar100-Alpha"


for iter in 0.1
do
    CUDA_VISIBLE_DEVICES=$device python run_baseline.py --yaml_name $yaml \
                                                        --dirichlet_alpha ${iter} \
                                                        --use_before_fc_emb 0 \
                                                        --local_lr 0.001 \
                                                        --logfile_info ${filename}${iter} \
                                                        --client_dataset "cifar100" \
                                                        --wandb_project "0116TheoreticalBound" \
                                                        --round 40 \
                                                        --select_client_num 15 \
                                                        --save_model_param 1
done