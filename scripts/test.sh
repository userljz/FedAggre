cd /home/ljz/FedAggre

#must config
device=7
yaml=test.yaml

CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                    --dirichlet_alpha 0.1 \
                                                    --use_before_fc_emb 1 \
                                                    --local_lr 0.001 \
                                                    --client_dataset "OrganAMNIST" \
                                                    --round 3



