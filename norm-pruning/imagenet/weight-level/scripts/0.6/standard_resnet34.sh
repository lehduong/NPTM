python prune.py --arch resnet34 \
--data ../../../data/image-net \
--save checkpoints/standard \
--percent 0.6 \
--pretrained  &&


python main_finetune.py \
--arch resnet34 \
--resume checkpoints/standard/pruned.pth.tar \
--save checkpoints/standard \
--data ../../../data/image-net \
--epochs 90 \
--wandb resnet_34_60_standard
