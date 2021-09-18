python vggprune.py \
--dataset cifar10 \
--model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/vgg_16.pt \
--save ../../../result/norm_pruning/filters/vgg16/standard &&

python main_finetune.py \
--refine ../../../result/norm_pruning/filters/vgg16/standard/pruned.pth.tar \
--save ../../../result/norm_pruning/filters/vgg16/standard \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--epochs 300 \
--wandb_name vgg_16_A_standard