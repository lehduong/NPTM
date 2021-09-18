python res34prune.py \
-v C \
--save checkpoints/llr \
--data ../../../data/image-net &&

python main_finetune.py \
--arch resnet34 \
--refine checkpoints/llr/pruned.pth.tar \
--save checkpoints/llr \
--data ../../../data/image-net \
--epochs 45 \
--use_llr \
--lr 0.1 \
--wandb resnet_34_C_llr