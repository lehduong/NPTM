prune_retrain_resnet56(){
    # train
    python train_test_split_main.py --dataset cifar100 \
    --arch resnet \
    --depth 56 \
    --save prune_retrain_checkpoints/resnet56_$2 \
    --seed $2 \
    --wandb resnet_56_standard_train_test_split &&
    
    # prune A
    python res56prune.py \
    --dataset cifar100 \
    -v A \
    --model prune_retrain_checkpoints/resnet56_$2/resnet_56_best.pt \
    --save prune_retrain_checkpoints/resnet56_$2 &&
    
    # onecycle
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/resnet56_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/resnet56_$2 \
    --dataset cifar100 \
    --arch resnet \
    --depth 56 \
    --epochs 40 \
    --use_onecycle \
    --seed $2 \
    --lr 0.1 \
    --wandb_name resnet_56_A_onecycle_40epochs_train_test_split &&
    
    # fine-tune
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/resnet56_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/resnet56_$2 \
    --dataset cifar100 \
    --arch resnet \
    --depth 56 \
    --epochs 40 \
    --seed $2 \
    --wandb_name resnet_56_A_finetune_40epochs_train_test_split
    
    
    # prune B
    python res56prune.py \
    --dataset cifar100 \
    -v B \
    --model prune_retrain_checkpoints/resnet56_$2/resnet_56_best.pt \
    --save prune_retrain_checkpoints/resnet56_$2 &&
    
    # onecycle
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/resnet56_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/resnet56_$2 \
    --dataset cifar100 \
    --arch resnet \
    --depth 56 \
    --epochs 40 \
    --use_onecycle \
    --seed $2 \
    --lr 0.1 \
    --wandb_name resnet_56_B_onecycle_40epochs_train_test_split &&
    
    # fine-tune
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/resnet56_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/resnet56_$2 \
    --dataset cifar100 \
    --arch resnet \
    --depth 56 \
    --epochs 40 \
    --seed $2 \
    --wandb_name resnet_56_B_finetune_40epochs_train_test_split
}

prune_retrain_resnet56 B 1 &&
prune_retrain_resnet56 B 2 &&
prune_retrain_resnet56 B 3 &&
prune_retrain_resnet56 B 4 &&
prune_retrain_resnet56 B 5