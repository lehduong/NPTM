python vggprune.py \
--dataset cifar100 \
-v B \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100/vgg_16.pt \
--save ../../../result/norm_pruning/filters/vgg16/standard &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/standard/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/standard \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--use_lr_rewind \
--epochs 152 \
--lr 0.1 \
--schedule 72 40 \
--wandb_name vgg_16_B_lr_rewind_152epochs