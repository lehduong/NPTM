# Prune 1
python res56prune.py \
--dataset cifar10 \
-v I \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/resnet_56.pt \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_1 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/prune_1/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_1 \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_56_I_1_onecycle &&

# Prune 2
python res56prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet56/onecycle/prune_1/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_2 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/prune_2/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_2 \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_56_I_2_onecycle &&

# Prune 3
python res56prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet56/onecycle/prune_2/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_3 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/prune_3/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_3 \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_56_I_3_onecycle &&

# Prune 4
python res56prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet56/onecycle/prune_3/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_4 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/prune_4/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_4 \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_56_I_4_onecycle &&

# Prune 5
python res56prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet56/onecycle/prune_4/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_5 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/prune_5/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_5 \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_56_I_5_onecycle &&

# Prune 6
python res56prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet56/onecycle/prune_5/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_6 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/prune_6/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_6 \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_56_I_6_onecycle &&

# Prune 7
python res56prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet56/onecycle/prune_6/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_7 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/prune_7/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_7 \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_56_I_7_onecycle &&

# Prune 8
python res56prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet56/onecycle/prune_7/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_8 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/prune_8/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/prune_8 \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_56_I_8_onecycle