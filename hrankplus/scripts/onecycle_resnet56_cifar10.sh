python main.py \
--job_dir ../result/hrankplus/resnet_56/onecycle \
--data_dir ../data \
--use_pretrain \
--pretrain_dir ../iclr2021_checkpoints/hrankplus/cifar/cifar10/resnet_56.pt \
--arch resnet_56 \
--compress_rate [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 \
--lr 0.1 \
--use_onecycle \
--epochs 300 \
--weight_decay 0.0005