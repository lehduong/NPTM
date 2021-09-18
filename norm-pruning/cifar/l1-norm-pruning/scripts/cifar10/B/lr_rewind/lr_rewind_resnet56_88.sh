python res56prune.py \
--dataset cifar10 \
-v B \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/resnet_56.pt \
--save ../../../result/norm_pruning/filters/resnet56/lr_rewind &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/lr_rewind/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/lr_rewind \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--use_lr_rewind \
--epochs 88 \
--wandb_name resnet_56_B_lr_rewind_88epochs