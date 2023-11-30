cd /home/ljz/FedAggre

#must config
device=7
yaml=train1.yaml
# yaml=test.yaml

CUDA_VISIBLE_DEVICES=$device python fedaggre.py --yaml_name $yaml  #&> ./log/test2


#------
# filename="1104_cifar100_lrtest_"
# for lr in 0.01 0.001 0.0001
# do
#     CUDA_VISIBLE_DEVICES=$device python fedmultidis.py --yaml_name $yaml --distill_lr $lr --logfile_info ${filename}${lr}
# done

# ------ 1122 ------
#device=3
#filename="1122_locallr_"
#for lr in 0.1 1.0 10.0
#do
#   CUDA_VISIBLE_DEVICES=$device python fedmultidis.py --yaml_name $yaml --local_lr $lr --logfile_info ${filename}${lr}
#done

