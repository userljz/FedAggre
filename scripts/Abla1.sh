cd /home/ljz/FedAggre

#must config
device=6
yaml=train1.yaml

filename="1228Abla-PseudoSampleNum-"


for geneN in 500 1000 5000 
do
    CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml \
                                                    --dirichlet_alpha 0.1 \
                                                    --use_before_fc_emb 1 \
                                                    --local_lr 0.005 \
                                                    --logfile_info ${filename}${geneN} \
                                                    --client_dataset "cifar100" \
                                                    --wandb_project "20231228Abla" \
                                                    --round 40 \
                                                    --gene_num $geneN
done
