#!/bin/zsh
#
# FileName: 	run_gpu
# CreatedDate:  2021-05-26 02:22:41 +0900
# LastModified: 2022-02-14 01:28:08 +0900
#


nohup python main.py ../datas ../outputs --device cuda:0 --dataset_name mycustom --n_epochs 100 --decay_epoch 80 --sample_interval 20 --checkpoint_interval 10 --batch_size 1 > results.log &

#export CUDA_VISIBLE_DEVICES=2,3
#nohup python main.py ../datas ../outputs --device cuda --multigpu --dataset_name mycustom --n_epochs 100 --decay_epoch 80 --sample_interval 20 --checkpoint_interval 20 --batch_size 2 > results.log &
#

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python main.py ../datas ../outputs --device cuda --multigpu --initial --dataset_name mycustom --n_epochs 100 --decay_epoch 80 --sample_interval 100 --checkpoint_interval 20 --batch_size 2 > results.log &


return


# test sample
nohup python test.py ../datas/IMG_0340 ../outputs/2021_06_25_14_24/green2cherry/hoge ../outputs/2021_06_25_14_24/saved_models/G_AB_0199.pth --device cuda:2 --img_height 512 --img_width 1024 --n_residual_blocks 13 &

