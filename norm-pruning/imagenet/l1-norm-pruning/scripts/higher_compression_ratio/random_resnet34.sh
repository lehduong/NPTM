python res34prune.py \
-v C \
--save checkpoints/random \
--random_rank \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet34 \
--refine checkpoints/random/pruned.pth.tar \
--save checkpoints/random \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_34_C_random