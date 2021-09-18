python res110prune.py \
--dataset cifar100 \
-v D \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100/resnet_110.pt \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind \
--dataset cifar100 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--epochs 88 \
--wandb_name resnet_110_D_lr_rewind_88epochs