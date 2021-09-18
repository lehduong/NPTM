python main.py \
--job_dir ../result/hrankplus/vgg_16_bn/standard \
--data_dir ../data \
--use_pretrain \
--pretrain_dir ../iclr2021_checkpoints/hrankplus/cifar/cifar10/vgg_16_bn.pt \
--arch vgg_16_bn \
--compress_rate [0.45]*7+[0.78]*5 \
--lr 0.01 \
--weight_decay 0.005 \
--lr_decay_step "50,100"
