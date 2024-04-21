cd /home/ljz/FedAggre

#must config
device=7
yaml=ours.yaml



CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                --dirichlet_alpha 0.01 \
                                                --use_before_fc_emb 1 \
                                                --local_lr 0.001 \
                                                --logfile_info "CANAL_CLIP_alpha0.01_meaningful_anchor_Local100" \
                                                --client_dataset "cifar100" \
                                                --wandb_project "20240419"


