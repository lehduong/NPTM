python res56prune.py \
--dataset cifar100 \
-v D \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100/resnet_56.pt \
--save ../../../result/norm_pruning/filters/resnet56/llr &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/llr/40epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 40 \
--use_llr \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_56_D_llr_40epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/llr/56epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 56 \
--use_llr \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_56_D_llr_56epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/llr/72epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 72 \
--use_llr \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_56_D_llr_72epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/llr/88epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 88 \
--use_llr \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_56_D_llr_88epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/llr/104epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 104 \
--use_llr \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_56_D_llr_104epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/llr/120epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 120 \
--use_llr \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_56_D_llr_120epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/llr/136epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 136 \
--use_llr \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_56_D_llr_136epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/llr/152epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 152 \
--use_llr \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_56_D_llr_152epochs