cd /home/ljz/FedAggre

#must config
device=3
yaml=train1.yaml


filename="0107Abla-ALBEF-base-CIFAR100-Alpha"


for iter in 0.1
do
    CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                    --dirichlet_alpha $iter \
                                                    --use_before_fc_emb 1 \
                                                    --local_lr 0.001 \
                                                    --logfile_info ${filename}${iter} \
                                                    --client_dataset "cifar100" \
                                                    --wandb_project "20240107" \
                                                    --round 40 \
                                                    --select_client_num 15 \
                                                    --model_name "ALBEF-base"
done
