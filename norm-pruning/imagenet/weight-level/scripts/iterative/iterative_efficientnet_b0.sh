# PRUNE 1
python prune.py --arch efficientnet_b0 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_1 \
--percent 0.2 \
--pretrained  &&


python efficientnet_finetune.py \
../../../data/image-net \
--model efficientnet_b0 \
--resume checkpoints/onecycle/prune_1/pruned.pth.tar \
--output checkpoints/onecycle/prune_1 \
--epoch 90 \
--use_onecycle \
--wandb iterative_efficientnet_b0_20_onecycle

# PRUNE 2
python prune.py --arch efficientnet_b0 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_2 \
--percent 0.3 \
--pretrained \
--resume checkpoints/onecycle/prune_1/finetune/last.pth.tar &&

python efficientnet_finetune.py \
../../../data/image-net \
--model efficientnet_b0 \
--resume checkpoints/onecycle/prune_2/pruned.pth.tar \
--output checkpoints/onecycle/prune_2 \
--epoch 90 \
--use_onecycle \
--wandb iterative_efficientnet_b0_30_onecycle

# PRUNE 3
python prune.py --arch efficientnet_b0 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_3 \
--percent 0.4 \
--pretrained \
--resume checkpoints/onecycle/prune_2/finetune/last.pth.tar &&

python efficientnet_finetune.py \
../../../data/image-net \
--model efficientnet_b0 \
--resume checkpoints/onecycle/prune_3/pruned.pth.tar \
--output checkpoints/onecycle/prune_3 \
--epoch 90 \
--use_onecycle \
--wandb iterative_efficientnet_b0_40_onecycle

# PRUNE 4
python prune.py --arch efficientnet_b0 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_4 \
--percent 0.5 \
--pretrained \
--resume checkpoints/onecycle/prune_3/finetune/last.pth.tar &&

python efficientnet_finetune.py \
../../../data/image-net \
--model efficientnet_b0 \
--resume checkpoints/onecycle/prune_4/pruned.pth.tar \
--output checkpoints/onecycle/prune_4 \
--epoch 90 \
--use_onecycle \
--wandb iterative_efficientnet_b0_50_onecycle

# PRUNE 5
python prune.py --arch efficientnet_b0 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_5 \
--percent 0.6 \
--pretrained \
--resume checkpoints/onecycle/prune_4/finetune/last.pth.tar &&

python efficientnet_finetune.py \
../../../data/image-net \
--model efficientnet_b0 \
--resume checkpoints/onecycle/prune_5/pruned.pth.tar \
--output checkpoints/onecycle/prune_5 \
--epoch 90 \
--use_onecycle \
--wandb iterative_efficientnet_b0_60_onecycle

# PRUNE 6
python prune.py --arch efficientnet_b0 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_6 \
--percent 0.7 \
--pretrained \
--resume checkpoints/onecycle/prune_5/finetune/last.pth.tar &&

python efficientnet_finetune.py \
../../../data/image-net \
--model efficientnet_b0 \
--resume checkpoints/onecycle/prune_6/pruned.pth.tar \
--output checkpoints/onecycle/prune_6 \
--epoch 90 \
--use_onecycle \
--wandb iterative_efficientnet_b0_70_onecycle

# PRUNE 7
python prune.py --arch efficientnet_b0 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_7 \
--percent 0.7 \
--pretrained \
--resume checkpoints/onecycle/prune_6/finetune/last.pth.tar &&

python efficientnet_finetune.py \
../../../data/image-net \
--model efficientnet_b0 \
--resume checkpoints/onecycle/prune_7/pruned.pth.tar \
--output checkpoints/onecycle/prune_7 \
--epoch 90 \
--use_onecycle \
--wandb iterative_efficientnet_b0_80_onecycle