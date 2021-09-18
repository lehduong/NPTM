python vggprune.py \
--dataset cifar10 \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/vgg_16.pt \
--save ../../../result/norm_pruning/filters/vgg16/llr  &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr/40epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--lr 0.1 \
--wandb_name vgg_16_A_llr_40epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr/56epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 56 \
--lr 0.1 \
--wandb_name vgg_16_A_llr_56epochs

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr/72epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 72 \
--lr 0.1 \
--wandb_name vgg_16_A_llr_72epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr/88epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 88 \
--lr 0.1 \
--wandb_name vgg_16_A_llr_88epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr/104epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 104 \
--lr 0.1 \
--wandb_name vgg_16_A_llr_104epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr/120epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 120 \
--lr 0.1 \
--wandb_name vgg_16_A_llr_120epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr/136epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 136 \
--lr 0.1 \
--wandb_name vgg_16_A_llr_136epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr/152epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 152 \
--lr 0.1 \
--wandb_name vgg_16_A_llr_152epochs