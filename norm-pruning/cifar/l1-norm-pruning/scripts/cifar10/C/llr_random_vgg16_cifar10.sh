python vggprune.py \
--dataset cifar10 \
-v B \
--random_rank \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/vgg_16.pt \
--save ../../../result/norm_pruning/filters/vgg16/llr_random  &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr_random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr_random/40epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--lr 0.1 \
--wandb_name vgg_16_B_llr_random_40epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr_random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr_random/56epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 56 \
--lr 0.1 \
--wandb_name vgg_16_B_llr_random_56epochs

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr_random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr_random/72epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 72 \
--lr 0.1 \
--wandb_name vgg_16_B_llr_random_72epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr_random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr_random/88epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 88 \
--lr 0.1 \
--wandb_name vgg_16_B_llr_random_88epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr_random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr_random/104epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 104 \
--lr 0.1 \
--wandb_name vgg_16_B_llr_random_104epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr_random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr_random/120epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 120 \
--lr 0.1 \
--wandb_name vgg_16_B_llr_random_120epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr_random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr_random/136epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 136 \
--lr 0.1 \
--wandb_name vgg_16_B_llr_random_136epochs &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/llr_random/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/llr_random/152epochs \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--use_llr \
--epochs 152 \
--lr 0.1 \
--wandb_name vgg_16_B_llr_random_152epochs