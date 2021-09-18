python res34prune.py \
-v D \
--save checkpoints/standard \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet34 \
--epochs 90 \
--refine checkpoints/standard/pruned.pth.tar \
--save checkpoints/standard \
--data ../../../data/image-net \
--wandb resnet_34_D_standard