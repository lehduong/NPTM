python prune.py --arch resnet34 \
--data ../../../data/image-net \
--save checkpoints/onecycle \
--percent 0.9 \
--pretrained  &&


python main_finetune.py \
--arch resnet34 \
--resume checkpoints/onecycle/pruned.pth.tar \
--save checkpoints/onecycle \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_34_90_onecycle
