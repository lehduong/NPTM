prune_retrain_resnet110(){
    # train
    python train_test_split_main.py --dataset cifar10 \
    --arch resnet \
    --depth 110 \
    --save prune_retrain_checkpoints/resnet110_$2 \
    --seed $2 \
    --wandb resnet_110_standard_train_test_split &&
    
    # prune A
    python res110prune.py \
    --dataset cifar10 \
    -v A \
    --model prune_retrain_checkpoints/resnet110_$2/resnet_110_best.pt \
    --save prune_retrain_checkpoints/resnet110_$2 &&
    
    # onecycle
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/resnet110_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/resnet110_$2 \
    --dataset cifar10 \
    --arch resnet \
    --depth 110 \
    --epochs 40 \
    --use_onecycle \
    --seed $2 \
    --lr 0.1 \
    --wandb_name resnet_110_A_onecycle_40epochs_train_test_split &&
    
    # fine-tune
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/resnet110_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/resnet110_$2 \
    --dataset cifar10 \
    --arch resnet \
    --depth 110 \
    --epochs 40 \
    --seed $2 \
    --wandb_name resnet_110_A_finetune_40epochs_train_test_split
    
    
    # prune B
    python res110prune.py \
    --dataset cifar10 \
    -v B \
    --model prune_retrain_checkpoints/resnet110_$2/resnet_110_best.pt \
    --save prune_retrain_checkpoints/resnet110_$2 &&
    
    # onecycle
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/resnet110_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/resnet110_$2 \
    --dataset cifar10 \
    --arch resnet \
    --depth 110 \
    --epochs 40 \
    --use_onecycle \
    --seed $2 \
    --lr 0.1 \
    --wandb_name resnet_110_B_onecycle_40epochs_train_test_split &&
    
    # fine-tune
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/resnet110_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/resnet110_$2 \
    --dataset cifar10 \
    --arch resnet \
    --depth 110 \
    --epochs 40 \
    --seed $2 \
    --wandb_name resnet_110_B_finetune_40epochs_train_test_split
}

prune_retrain_resnet110 B 1 &&
prune_retrain_resnet110 B 2 &&
prune_retrain_resnet110 B 3 &&
prune_retrain_resnet110 B 4 &&
prune_retrain_resnet110 B 5