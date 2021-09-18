python main.py \
--job_dir ../result/hrank/densenet_40/standard \
--data_dir ../data \
--epochs 30 \
--resume ../iclr2021_checkpoints/hrank/cifar/cifar10/densenet_40.pt \
--adjust_prune_ckpt \
--arch densenet_40 \
--compress_rate [0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0] \
--gpu 0