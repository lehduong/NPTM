# Prune 1
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar1000/resnet_110.pt \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_1 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/prune_1/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_1 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_1_onecycle &&

# Prune 2
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/onecycle/prune_1/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_2 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/prune_2/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_2 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_2_onecycle &&

# Prune 3
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/onecycle/prune_2/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_3 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/prune_3/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_3 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_3_onecycle &&

# Prune 4
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/onecycle/prune_3/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_4 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/prune_4/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_4 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_4_onecycle &&

# Prune 5
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/onecycle/prune_4/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_5 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/prune_5/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_5 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_5_onecycle &&

# Prune 6
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/onecycle/prune_5/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_6 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/prune_6/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_6 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_6_onecycle &&

# Prune 7
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/onecycle/prune_6/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_7 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/prune_7/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_7 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_7_onecycle &&

# Prune 8
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/onecycle/prune_7/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_8 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/prune_8/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/prune_8 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_onecycle \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_8_onecycle