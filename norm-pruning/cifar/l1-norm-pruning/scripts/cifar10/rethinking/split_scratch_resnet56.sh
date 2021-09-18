scratch_resnet56(){
    python train_test_split_main.py --dataset cifar10 \
    --arch resnet \
    --depth 56 \
    --save scratch_checkpoints/resnet56_$2 \
    --epochs 1 \
    --wandb_name redundant &&
    
    python res56prune.py \
    --dataset cifar10 \
    -v $1 \
    --model scratch_checkpoints/resnet56_$2/resnet_56_best.pt \
    --save scratch_checkpoints/resnet56_$2 &&
    
    python train_test_split_main_B.py \
    --scratch scratch_checkpoints/resnet56_$2/pruned.pth.tar \
    --save scratch_checkpoints/resnet56_$2 \
    --dataset cifar10 \
    --arch resnet \
    --depth 56 \
    --seed $2 \
    --wandb_name resnet_56_${1}_scratch_train_test_split
}

scratch_resnet56 A 1 &&
scratch_resnet56 B 1 &&
scratch_resnet56 A 2 &&
scratch_resnet56 B 2 &&
scratch_resnet56 A 3 &&
scratch_resnet56 B 3 &&
scratch_resnet56 A 4 &&
scratch_resnet56 B 4 &&
scratch_resnet56 A 5 &&
scratch_resnet56 B 5