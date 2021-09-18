python res56prune.py \
--dataset cifar10 \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/resnet_56.pt \
--save ../../../result/norm_pruning/filters/resnet56/standard &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/standard/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/standard \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--epochs 300 \
--wandb_name resnet_56_B_standard