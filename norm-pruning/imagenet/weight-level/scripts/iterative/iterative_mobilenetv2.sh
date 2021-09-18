# PRUNE 1
python prune.py --arch mobilenet_v2 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_1 \
--percent 0.2 \
--pretrained  &&


python main_finetune.py \
--arch mobilenet_v2 \
--resume checkpoints/onecycle/prune_1/pruned.pth.tar \
--save checkpoints/onecycle/prune_1 \
--data ../../../data/image-net \
--epochs 75 \
--use_onecycle \
--lr 0.045 \
--wd 0.00004 \
--div_factor 450 \
--wandb iterative_mobilenet_v2_20_onecycle

# PRUNE 2
python prune.py --arch mobilenet_v2 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_2 \
--percent 0.3 \
--pretrained \
--resume checkpoints/onecycle/prune_1/finetuned.pth.tar &&


python main_finetune.py \
--arch mobilenet_v2 \
--resume checkpoints/onecycle/prune_2/pruned.pth.tar \
--save checkpoints/onecycle/prune_2 \
--data ../../../data/image-net \
--epochs 75 \
--use_onecycle \
--lr 0.045 \
--wd 0.00004 \
--div_factor 450 \
--wandb iterative_mobilenet_v2_30_onecycle

# PRUNE 3
python prune.py --arch mobilenet_v2 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_3 \
--percent 0.4 \
--pretrained \
--resume checkpoints/onecycle/prune_2/finetuned.pth.tar &&


python main_finetune.py \
--arch mobilenet_v2 \
--resume checkpoints/onecycle/prune_3/pruned.pth.tar \
--save checkpoints/onecycle/prune_3 \
--data ../../../data/image-net \
--epochs 75 \
--use_onecycle \
--lr 0.045 \
--wd 0.00004 \
--div_factor 450 \
--wandb iterative_mobilenet_v2_40_onecycle

# PRUNE 4
python prune.py --arch mobilenet_v2 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_4 \
--percent 0.5 \
--pretrained \
--resume checkpoints/onecycle/prune_3/finetuned.pth.tar &&


python main_finetune.py \
--arch mobilenet_v2 \
--resume checkpoints/onecycle/prune_4/pruned.pth.tar \
--save checkpoints/onecycle/prune_4 \
--data ../../../data/image-net \
--epochs 75 \
--use_onecycle \
--lr 0.045 \
--wd 0.00004 \
--div_factor 450 \
--wandb iterative_mobilenet_v2_50_onecycle

# PRUNE 5
python prune.py --arch mobilenet_v2 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_5 \
--percent 0.6 \
--pretrained \
--resume checkpoints/onecycle/prune_4/finetuned.pth.tar &&


python main_finetune.py \
--arch mobilenet_v2 \
--resume checkpoints/onecycle/prune_5/pruned.pth.tar \
--save checkpoints/onecycle/prune_5 \
--data ../../../data/image-net \
--epochs 75 \
--use_onecycle \
--lr 0.045 \
--wd 0.00004 \
--div_factor 450 \
--wandb iterative_mobilenet_v2_60_onecycle

# PRUNE 6
python prune.py --arch mobilenet_v2 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_6 \
--percent 0.7 \
--pretrained \
--resume checkpoints/onecycle/prune_5/finetuned.pth.tar &&


python main_finetune.py \
--arch mobilenet_v2 \
--resume checkpoints/onecycle/prune_6/pruned.pth.tar \
--save checkpoints/onecycle/prune_6 \
--data ../../../data/image-net \
--epochs 75 \
--use_onecycle \
--lr 0.045 \
--wd 0.00004 \
--div_factor 450 \
--wandb iterative_mobilenet_v2_70_onecycle