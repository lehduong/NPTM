python vggprune.py \
--dataset cifar10 \
-v C \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/vgg_16.pt \
--save ../../../result/norm_pruning/filters/vgg16/onecycle  &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/onecycle/40epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_onecycle \
--lr 0.1 \
--wandb_name vgg_16_C_onecycle_40epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/onecycle/56epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_onecycle \
--epochs 56 \
--lr 0.1 \
--wandb_name vgg_16_C_onecycle_56epochs

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/onecycle/72epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_onecycle \
--epochs 72 \
--lr 0.1 \
--wandb_name vgg_16_C_onecycle_72epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/onecycle/88epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_onecycle \
--epochs 88 \
--lr 0.1 \
--wandb_name vgg_16_C_onecycle_88epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/onecycle/104epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_onecycle \
--epochs 104 \
--lr 0.1 \
--wandb_name vgg_16_C_onecycle_104epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/onecycle/120epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_onecycle \
--epochs 120 \
--lr 0.1 \
--wandb_name vgg_16_C_onecycle_120epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/onecycle/136epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_onecycle \
--epochs 136 \
--lr 0.1 \
--wandb_name vgg_16_C_onecycle_136epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/onecycle/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/onecycle/152epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_onecycle \
--epochs 152 \
--lr 0.1 \
--wandb_name vgg_16_C_onecycle_152epochs