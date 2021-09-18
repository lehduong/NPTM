# PRUNE 1
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_1 \
--percent 0.2 \
--prune_linear \
--pretrained  &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_1/pruned.pth.tar \
--save checkpoints/onecycle/prune_1 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
iterative_resnet50_prune_all_param_20_onecycle

# PRUNE 2
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_2 \
--percent 0.4 \
--prune_linear \
--resume checkpoints/onecycle/prune_1/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_2/pruned.pth.tar \
--save checkpoints/onecycle/prune_2 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
iterative_resnet50_prune_all_param_40_onecycle

# PRUNE 3
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_3 \
--percent 0.6 \
--prune_linear \
--resume checkpoints/onecycle/prune_2/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_3/pruned.pth.tar \
--save checkpoints/onecycle/prune_3 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
iterative_resnet50_prune_all_param_60_onecycle

# PRUNE 4
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_4 \
--percent 0.8 \
--prune_linear \
--resume checkpoints/onecycle/prune_3/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_4/pruned.pth.tar \
--save checkpoints/onecycle/prune_4 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
iterative_resnet50_prune_all_param_80_onecycle

# PRUNE 5
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_5 \
--percent 0.9 \
--prune_linear
--resume checkpoints/onecycle/prune_4/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_5/pruned.pth.tar \
--save checkpoints/onecycle/prune_5 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
iterative_resnet50_prune_all_param_90_onecycle

# PRUNE 6
python prune.py --arch resnet50 \
--data ../../../data/image-net \
--save checkpoints/onecycle/prune_6 \
--percent 0.95 \
--prune_linear \
--resume checkpoints/onecycle/prune_5/finetuned_best.pth.tar &&


python main_finetune.py \
--arch resnet50 \
--resume checkpoints/onecycle/prune_6/pruned.pth.tar \
--save checkpoints/onecycle/prune_6 \
--data ../../../data/image-net \
--epochs 45 \
--use_onecycle \
--lr 0.1 \
iterative_resnet50_prune_all_param_95_onecycle