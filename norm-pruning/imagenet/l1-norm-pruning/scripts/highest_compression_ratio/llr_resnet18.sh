python res18prune.py \
-v D \
--save checkpoints/llr \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet18 \
--refine checkpoints/llr/pruned.pth.tar \
--save checkpoints/llr \
--data ../../../data/image-net \
--epochs 45 \
--use_llr \
--lr 0.1 \
--wandb resnet_18_D_llr