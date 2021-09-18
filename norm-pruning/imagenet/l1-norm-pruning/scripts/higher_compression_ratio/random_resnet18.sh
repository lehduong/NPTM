python res18prune.py \
-v C \
--save checkpoints/random \
--random_rank \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet18 \
--refine checkpoints/random/pruned.pth.tar \
--save checkpoints/random \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_18_C_random