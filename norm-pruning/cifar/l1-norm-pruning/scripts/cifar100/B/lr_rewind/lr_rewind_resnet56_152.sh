python res56prune.py \
--dataset cifar100 \
-v B \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100/resnet_56.pt \
--save ../../../result/norm_pruning/filters/resnet56/lr_rewind &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/lr_rewind/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/lr_rewind \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--use_lr_rewind \
--epochs 152 \
--wandb_name resnet_56_B_lr_rewind_152epochs