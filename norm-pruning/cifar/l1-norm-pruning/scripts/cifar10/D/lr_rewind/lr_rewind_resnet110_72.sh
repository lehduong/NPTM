python res110prune.py \
--dataset cifar10 \
-v D \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/resnet_110.pt \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--epochs 72 \
--wandb_name resnet_110_D_lr_rewind_72epochs