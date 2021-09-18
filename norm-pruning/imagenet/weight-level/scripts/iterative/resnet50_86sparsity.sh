# PRUNE 1
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_1 \
--percent 0.2 \
--pretrained  &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_1/pruned.pth.tar \
--save checkpoints/onecycle/prune_1 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_50_86sparsity_1_onecycle

# PRUNE 2
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_2 \
--percent 0.4 \
--pretrained \
--resume checkpoints/onecycle/prune_1/finetuned.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_2/pruned.pth.tar \
--save checkpoints/onecycle/prune_2 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_50_86sparsity_2_onecycle

# PRUNE 3
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_3 \
--percent 0.6 \
--pretrained \
--resume checkpoints/onecycle/prune_2/finetuned.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_3/pruned.pth.tar \
--save checkpoints/onecycle/prune_3 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_50_86sparsity_3_onecycle

# PRUNE 4
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_4 \
--percent 0.8 \
--pretrained \
--resume checkpoints/onecycle/prune_3/finetuned.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_4/pruned.pth.tar \
--save checkpoints/onecycle/prune_4 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_50_86sparsity_4_onecycle

# PRUNE 5
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_5 \
--percent 0.86 \
--pretrained \
--resume checkpoints/onecycle/prune_4/finetuned.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_5/pruned.pth.tar \
--save checkpoints/onecycle/prune_5 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_50_86sparsity_5_onecycle

# PRUNE 6
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_6 \
--percent 0.9 \
--pretrained \
--resume checkpoints/onecycle/prune_5/finetuned.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_6/pruned.pth.tar \
--save checkpoints/onecycle/prune_6 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_50_86sparsity_6_onecycle