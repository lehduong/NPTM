python res34prune.py \
-v B \
--save checkpoints/lr_rewind \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet34 \
--refine checkpoints/lr_rewind/pruned.pth.tar \
--save checkpoints/lr_rewind \
--data ../../../data/image-net \
--epochs 45 \
--use_lr_rewind \
--lr 0.001 \
--wandb resnet_34_B_lr_rewind