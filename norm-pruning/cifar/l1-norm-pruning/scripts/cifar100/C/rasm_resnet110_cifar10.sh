python res110prune.py \
--dataset cifar100 \
-v C \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100/resnet_110.pt \
--save ../../../result/norm_pruning/filters/resnet110/randomsmall \
--random_rank &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/randomsmall/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/randomsmall \
--dataset cifar100 \
--arch resnet \
--depth 110 \
--epochs 300 \
--wandb_name resnet_110_C_randomsmall