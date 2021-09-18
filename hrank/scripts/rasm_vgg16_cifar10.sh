python main.py \
--job_dir ../result/hrank/vgg_16_bn/random_small \
--data_dir ../data \
--epochs 30 \
--resume ../iclr2021_checkpoints/hrank/cifar/cifar10/vgg_16_bn.pt \
--adjust_prune_ckpt \
--arch vgg_16_bn \
--compress_rate [0.95]+[0.5]*6+[0.9]*4+[0.8]*2 \
--random_rank \
--gpu 0