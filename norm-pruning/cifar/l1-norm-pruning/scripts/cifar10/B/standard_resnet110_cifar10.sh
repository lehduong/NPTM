python res110prune.py \
--dataset cifar10 \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/resnet_110.pt \
--save ../../../result/norm_pruning/filters/resnet110/standard &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/standard/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--epochs 300 \
--wandb_name resnet_110_B_standard