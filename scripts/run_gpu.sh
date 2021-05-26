#!/bin/zsh
#
# FileName: 	run_gpu
# CreatedDate:  2021-05-26 02:22:41 +0900
# LastModified: 2021-05-26 02:24:51 +0900
#



return
#!/bin/zsh
#
# FileName: 	run_cpu
# CreatedDate:  2021-05-26 02:21:05 +0900
# LastModified: 2021-05-26 02:24:51 +0900
#


nohup python main.py ../datas ../outputs --device cuda:1 --dataset_name mycustom --n_epochs 100 --decay_epoch 80 --epoch_interval 50 --checkpoint_interval 40 > hoge.log &

return
