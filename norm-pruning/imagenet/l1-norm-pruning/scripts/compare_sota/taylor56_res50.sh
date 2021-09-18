python residual_prune.py --save checkpoints/taylor52 \
--arch resnet50 \
--data ../../../data/image-net \
-v Taylor52 &&

python main_finetune.py \
--arch resnet50 \
--refine checkpoints/taylor52/pruned.pth.tar \
--save checkpoints/taylor52 \
--data ../../../data/image-net \
--epochs 25 \
--use_onecycle \
--lr 0.01 \
--wandb resnet_50_Taylor52_onecycle_0.01