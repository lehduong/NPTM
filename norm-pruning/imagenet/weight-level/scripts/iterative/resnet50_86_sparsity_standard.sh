# PRUNE 1
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/standard/prune_1 \
--percent 0.2 \
--pretrained  &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/standard/prune_1/pruned.pth.tar \
--save checkpoints/standard/prune_1 \
--data ../../../data/image-net \
--epochs 45 \
--wandb resnet_50_86sparsity_1_standard

# PRUNE 2
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/standard/prune_2 \
--percent 0.4 \
--pretrained \
--resume checkpoints/standard/prune_1/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/standard/prune_2/pruned.pth.tar \
--save checkpoints/standard/prune_2 \
--data ../../../data/image-net \
--epochs 45 \
--wandb resnet_50_86sparsity_2_standard

# PRUNE 3
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/standard/prune_3 \
--percent 0.6 \
--pretrained \
--resume checkpoints/standard/prune_2/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/standard/prune_3/pruned.pth.tar \
--save checkpoints/standard/prune_3 \
--data ../../../data/image-net \
--epochs 45 \
--wandb resnet_50_86sparsity_3_standard

# PRUNE 4
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/standard/prune_4 \
--percent 0.8 \
--pretrained \
--resume checkpoints/standard/prune_3/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/standard/prune_4/pruned.pth.tar \
--save checkpoints/standard/prune_4 \
--data ../../../data/image-net \
--epochs 45 \
--wandb resnet_50_86sparsity_4_standard

# PRUNE 5
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/standard/prune_5 \
--percent 0.86 \
--pretrained \
--resume checkpoints/standard/prune_4/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/standard/prune_5/pruned.pth.tar \
--save checkpoints/standard/prune_5 \
--data ../../../data/image-net \
--epochs 45 \
--wandb resnet_50_86sparsity_5_standard

# PRUNE 6
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/standard/prune_6 \
--percent 0.9 \
--pretrained \
--resume checkpoints/standard/prune_5/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/standard/prune_6/pruned.pth.tar \
--save checkpoints/standard/prune_6 \
--data ../../../data/image-net \
--epochs 45 \
--wandb resnet_50_86sparsity_6_standard