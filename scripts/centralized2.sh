cd /home/ljz/FedAggre

#must config
device=6
yaml=centralized.yaml

filename="0106-OrganAMNIST-centralized"



CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                --dirichlet_alpha 0 \
                                                --use_before_fc_emb 0 \
                                                --local_lr 0.001 \
                                                --logfile_info ${filename}${iter} \
                                                --client_dataset "OrganAMNIST" \
                                                --wandb_project "0106Result" \
                                                --round 40 \
                                                --select_client_num 0

