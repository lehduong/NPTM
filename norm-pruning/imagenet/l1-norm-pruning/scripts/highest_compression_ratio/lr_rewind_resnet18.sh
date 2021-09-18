python res18prune.py \
-v D \
--save checkpoints/lr_rewind \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet18 \
--refine checkpoints/lr_rewind/pruned.pth.tar \
--save checkpoints/lr_rewind \
--data ../../../data/image-net \
--epochs 45 \
--use_lr_rewind \
--lr 0.001 \
--wandb resnet_18_D_lr_rewind