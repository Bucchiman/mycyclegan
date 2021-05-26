#!/bin/zsh
#
# FileName: 	run_cpu
# CreatedDate:  2021-05-26 02:21:05 +0900
# LastModified: 2021-05-26 02:26:28 +0900
#


cd ../src
python main.py ../datas ../outputs --device cpu --dataset_name mycustom --n_epochs 100 --decay_epoch 80 --checkpoint_interval 40

return
