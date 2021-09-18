# Prune 1
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar1000/resnet_110.pt \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_1 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_1/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_1 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--lr 0.001 \
--wandb_name iterative_resnet_110_I_1_lr_rewind &&

# Prune 2
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_1/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_2 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_2/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_2 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--lr 0.001 \
--wandb_name iterative_resnet_110_I_2_lr_rewind &&

# Prune 3
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_2/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_3 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_3/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_3 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--lr 0.001 \
--wandb_name iterative_resnet_110_I_3_lr_rewind &&

# Prune 4
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_3/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_4 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_4/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_4 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--lr 0.001 \
--wandb_name iterative_resnet_110_I_4_lr_rewind &&

# Prune 5
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_4/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_5 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_5/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_5 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--lr 0.001 \
--wandb_name iterative_resnet_110_I_5_lr_rewind &&

# Prune 6
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_5/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_6 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_6/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_6 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--lr 0.001 \
--wandb_name iterative_resnet_110_I_6_lr_rewind &&

# Prune 7
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_6/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_7 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_7/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_7 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--lr 0.001 \
--wandb_name iterative_resnet_110_I_7_lr_rewind &&

# Prune 8
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_7/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_8 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_8/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/lr_rewind/prune_8 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_lr_rewind \
--lr 0.001 \
--wandb_name iterative_resnet_110_I_8_lr_rewind