python res18prune.py \
-v D \
--save checkpoints/standard \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet18 \
--epochs 90 \
--refine checkpoints/standard/pruned.pth.tar \
--save checkpoints/standard \
--data ../../../data/image-net \
--wandb resnet_18_D_standard