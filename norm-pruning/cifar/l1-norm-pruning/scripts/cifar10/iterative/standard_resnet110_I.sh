# Prune 1
python res110prune.py \
--dataset cifar10 \
-v I \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/resnet_110.pt \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_1 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/standard/prune_1/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_1 \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--wandb_name iterative_resnet_110_I_1_standard &&

# Prune 2
python res110prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/standard/prune_1/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_2 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/standard/prune_2/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_2 \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--wandb_name iterative_resnet_110_I_2_standard &&

# Prune 3
python res110prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/standard/prune_2/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_3 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/standard/prune_3/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_3 \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--wandb_name iterative_resnet_110_I_3_standard &&

# Prune 4
python res110prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/standard/prune_3/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_4 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/standard/prune_4/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_4 \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--wandb_name iterative_resnet_110_I_4_standard &&

# Prune 5
python res110prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/standard/prune_4/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_5 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/standard/prune_5/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_5 \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--wandb_name iterative_resnet_110_I_5_standard &&

# Prune 6
python res110prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/standard/prune_5/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_6 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/standard/prune_6/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_6 \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--wandb_name iterative_resnet_110_I_6_standard &&

# Prune 7
python res110prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/standard/prune_6/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_7 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/standard/prune_7/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_7 \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--wandb_name iterative_resnet_110_I_7_standard &&

# Prune 8
python res110prune.py \
--dataset cifar10 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/standard/prune_7/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_8 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/standard/prune_8/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/standard/prune_8 \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--wandb_name iterative_resnet_110_I_8_standard