cd /home/ljz/FedAggre

#must config
device=3
yaml=test.yaml

CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                    --dirichlet_alpha 0.1 \
                                                    --use_before_fc_emb 1 \
                                                    --local_lr 0.001 \
                                                    --round 3



