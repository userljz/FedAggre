cd /home/ljz/FedAggre

#must config
device=0
yaml=baseline.yaml
# yaml=test.yaml
# yaml=use_extra_emb.yaml

CUDA_VISIBLE_DEVICES=$device python run_baseline.py --yaml_name $yaml  #&> ./log/test