cd /home/ljz/FedAggre

#must config
device=5
yaml=train1.yaml

# CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml  #&> ./log/test


filename="1228LrGridSearch-PathMNIST-"


for lr in 0.001 0.01 0.1
do
    CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                    --dirichlet_alpha 0.1 \
                                                    --use_before_fc_emb 1 \
                                                    --local_lr $lr \
                                                    --logfile_info ${filename}${lr} \
                                                    --client_dataset "PathMNIST" \
                                                    --wandb_project "20231228LrGridSearch" \
                                                    --round 20
done
