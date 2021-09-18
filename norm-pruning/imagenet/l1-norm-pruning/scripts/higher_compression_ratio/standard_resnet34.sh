python res34prune.py \
-v C \
--save checkpoints/standard \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet34 \
--epochs 90 \
--refine checkpoints/standard/pruned.pth.tar \
--save checkpoints/standard \
--data ../../../data/image-net \
--wandb resnet_34_C_standard