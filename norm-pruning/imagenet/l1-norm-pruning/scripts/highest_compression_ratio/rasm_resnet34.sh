python res34prune.py \
-v D \
--save checkpoints/rasm \
--random_rank \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet34 \
--epochs 90 \
--refine checkpoints/rasm/pruned.pth.tar \
--save checkpoints/rasm \
--data ../../../data/image-net \
--wandb resnet_34_D_rasm