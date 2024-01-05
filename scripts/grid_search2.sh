cd /home/ljz/FedAggre

#must config
device=7
yaml=train1.yaml

filename="1228Result-emnist62-Alpha"


for iter in 0.01 0.05 0.1 0.5 1
do
    CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                    --dirichlet_alpha ${iter} \
                                                    --use_before_fc_emb 1 \
                                                    --local_lr 0.001 \
                                                    --logfile_info ${filename}${iter} \
                                                    --client_dataset "emnist62" \
                                                    --wandb_project "1228Result" \
                                                    --round 40 \
                                                    --select_client_num 15
done
