python res110prune.py \
--dataset cifar10 \
-v C \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/resnet_110.pt \
--save ../../../result/norm_pruning/filters/resnet110/onecycle &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/40epochs \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--epochs 40 \
--use_onecycle \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_110_C_onecycle_40epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/56epochs \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--epochs 56 \
--use_onecycle \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_110_C_onecycle_56epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/72epochs \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--epochs 72 \
--use_onecycle \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_110_C_onecycle_72epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/88epochs \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--epochs 88 \
--use_onecycle \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_110_C_onecycle_88epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/104epochs \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--epochs 104 \
--use_onecycle \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_110_C_onecycle_104epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/120epochs \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--epochs 120 \
--use_onecycle \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_110_C_onecycle_120epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/136epochs \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--epochs 136 \
--use_onecycle \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_110_C_onecycle_136epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/resnet110/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/resnet110/onecycle/152epochs \
--dataset cifar10 \
--arch resnet \
--depth 110 \
--epochs 152 \
--use_onecycle \
--init_lr 0.001 \
--lr 0.1 \
--wandb_name resnet_110_C_onecycle_152epochs