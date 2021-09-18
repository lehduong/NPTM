python res18prune.py \
-v D \
--save checkpoints/onecycle \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet18 \
--refine checkpoints/onecycle/pruned.pth.tar \
--save checkpoints/onecycle \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_18_D_onecycle