#!/bin/zsh
#
# FileName: 	run_gpu
# CreatedDate:  2021-05-26 02:22:41 +0900
# LastModified: 2021-05-28 20:57:57 +0900
#


#nohup python main.py ../datas ../outputs --device cuda:1 --dataset_name mycustom --n_epochs 100 --decay_epoch 80 --sample_interval 50 --checkpoint_interval 40 > hoge.log &

cd ../src
export CUDA_VISIBLE_DEVICES=2,3
nohup python main.py ../datas ../outputs --device cuda --multigpu --dataset_name mycustom --n_epochs 100 --decay_epoch 80 --sample_interval 20 --checkpoint_interval 20 --batch_size 2 > results.log &

return
