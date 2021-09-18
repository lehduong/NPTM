# Prune 1
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar1000/resnet_110.pt \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_1 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/llr/prune_1/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_1 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_llr \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_1_llr &&

# Prune 2
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/llr/prune_1/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_2 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/llr/prune_2/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_2 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_llr \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_2_llr &&

# Prune 3
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/llr/prune_2/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_3 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/llr/prune_3/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_3 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_llr \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_3_llr &&

# Prune 4
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/llr/prune_3/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_4 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/llr/prune_4/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_4 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_llr \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_4_llr &&

# Prune 5
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/llr/prune_4/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_5 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/llr/prune_5/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_5 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_llr \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_5_llr &&

# Prune 6
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/llr/prune_5/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_6 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/llr/prune_6/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_6 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_llr \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_6_llr &&

# Prune 7
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/llr/prune_6/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_7 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/llr/prune_7/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_7 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_llr \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_7_llr &&

# Prune 8
python res110prune.py \
--dataset cifar1000 \
-v I \
--model ../../../result/norm_pruning/filters/resnet110/llr/prune_7/checkpoint.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_8 &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/llr/prune_8/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/llr/prune_8 \
--dataset cifar1000 \
--arch resnet \
--depth 110 \
--use_llr \
--lr 0.1 \
--wandb_name iterative_resnet_110_I_8_llr