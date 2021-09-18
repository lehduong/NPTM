python main.py \
--job_dir ../result/hrank/resnet_110/random_small \
--data_dir ../data \
--epochs 30 \
--resume ../iclr2021_checkpoints/hrank/cifar/cifar10/resnet_110.pt \
--adjust_prune_ckpt \
--arch resnet_110 \
--compress_rate [0.1]+[0.40]*36+[0.40]*36+[0.4]*36 \
--random_rank \
--gpu 0