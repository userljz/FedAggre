cd /home/ljz/FedMultiDis

#must config
device=3
yaml=only_st.yaml

CUDA_VISIBLE_DEVICES=$device python fedmultidis.py --yaml_name $yaml  #&> ./log/test


#------
# filename="1104_cifar100_lrtest_"
# for lr in 0.01 0.001 0.0001
# do
#     CUDA_VISIBLE_DEVICES=$device python fedmultidis.py --yaml_name $yaml --distill_lr $lr --logfile_info ${filename}${lr}
# done

