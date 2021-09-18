python vggprune.py \
--dataset cifar10 \
-v C \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/vgg_16.pt \
--save ../../../result/norm_pruning/filters/vgg16/randomsmall \
--random_rank &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/randomsmall/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/randomsmall \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--epochs 300 \
--wandb_name vgg_16_C_randomsmall