python main.py \
--job_dir ../result/hrank/resnet_56/random_small \
--data_dir ../data \
--epochs 30 \
--resume ../iclr2021_checkpoints/hrank/cifar/cifar10/resnet_56.pt \
--adjust_prune_ckpt \
--arch resnet_56 \
--compress_rate [0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4] \
--random_rank \
--gpu 0