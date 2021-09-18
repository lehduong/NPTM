python res18prune.py \
-v D \
--save checkpoints/rasm \
--random_rank \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet18 \
--epochs 90 \
--refine checkpoints/rasm/pruned.pth.tar \
--save checkpoints/rasm \
--data ../../../data/image-net \
--wandb resnet_18_D_rasm