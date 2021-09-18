python res56prune.py \
--dataset cifar100 \
-v B \
--random_rank \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100/resnet_56.pt \
--save ../../../result/norm_pruning/filters/resnet56/random &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/random/40epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 40 \
--use_random \
--lr 0.1 \
--wandb_name resnet_56_B_random_40epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/random/56epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 56 \
--use_random \
--lr 0.1 \
--wandb_name resnet_56_B_random_56epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/random/72epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 72 \
--use_random \
--lr 0.1 \
--wandb_name resnet_56_B_random_72epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/random/88epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 88 \
--use_random \
--lr 0.1 \
--wandb_name resnet_56_B_random_88epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/random/104epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 104 \
--use_random \
--lr 0.1 \
--wandb_name resnet_56_B_random_104epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/random/120epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 120 \
--use_random \
--lr 0.1 \
--wandb_name resnet_56_B_random_120epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/random/136epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 136 \
--use_random \
--lr 0.1 \
--wandb_name resnet_56_B_random_136epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/random/152epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 152 \
--use_random \
--lr 0.1 \
--wandb_name resnet_56_B_random_152epochs