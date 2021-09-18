python res56prune.py \
--dataset cifar100 \
-v C \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100/resnet_56.pt \
--save ../../../result/norm_pruning/filters/resnet56/onecycle &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/40epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 40 \
--use_onecycle \
--lr 0.1 \
--wandb_name resnet_56_C_onecycle_40epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/56epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 56 \
--use_onecycle \
--lr 0.1 \
--wandb_name resnet_56_C_onecycle_56epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/72epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 72 \
--use_onecycle \
--lr 0.1 \
--wandb_name resnet_56_C_onecycle_72epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/88epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 88 \
--use_onecycle \
--lr 0.1 \
--wandb_name resnet_56_C_onecycle_88epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/104epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 104 \
--use_onecycle \
--lr 0.1 \
--wandb_name resnet_56_C_onecycle_104epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/120epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 120 \
--use_onecycle \
--lr 0.1 \
--wandb_name resnet_56_C_onecycle_120epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/136epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 136 \
--use_onecycle \
--lr 0.1 \
--wandb_name resnet_56_C_onecycle_136epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet56/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet56/onecycle/152epochs \
--dataset cifar100 \
--arch resnet \
--depth 56 \
--epochs 152 \
--use_onecycle \
--lr 0.1 \
--wandb_name resnet_56_C_onecycle_152epochs