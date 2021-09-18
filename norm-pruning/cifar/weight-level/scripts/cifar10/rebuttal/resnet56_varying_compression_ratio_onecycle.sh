
prune(){
    python cifar_prune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar10 \
    --percent $3 \
    --resume ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar10/$1_$2.pt \
    --save_dir ../../../result/norm_pruning/weights/$1$2/onecycle
}


random_prune(){
    python cifar_prune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar10 \
    --percent $3 \
    --random_rank \
    --resume ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar10/$1_$2.pt \
    --save_dir ../../../result/norm_pruning/weights/$1$2/random
}

retrain(){
    python cifar_finetune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar10 \
    --resume ../../../result/norm_pruning/weights/$1$2/onecycle/pruned.pth.tar \
    --save_dir ../../../result/norm_pruning/weights/$1$2/onecycle/$4epochs \
    --lr $3 \
    --epochs $4 \
    --use_onecycle \
    --wandb_name $1_$2_$5_onecycle_$4epochs
}

random_retrain(){
    python cifar_finetune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar10 \
    --resume ../../../result/norm_pruning/weights/$1$2/random/pruned.pth.tar \
    --save_dir ../../../result/norm_pruning/weights/$1$2/random/$4epochs \
    --lr $3 \
    --epochs $4 \
    --use_onecycle \
    --wandb_name $1_$2_$5_random_$4epochs
}

resnet56(){
    prune resnet 56 0.2
    retrain resnet 56 0.1 40 0.2
    prune resnet 56 0.3
    retrain resnet 56 0.1 40 0.3
    prune resnet 56 0.4
    retrain resnet 56 0.1 40 0.4
    prune resnet 56 0.5
    retrain resnet 56 0.1 40 0.5
    prune resnet 56 0.6
    retrain resnet 56 0.1 40 0.6
    prune resnet 56 0.7
    retrain resnet 56 0.1 40 0.7
    prune resnet 56 0.8
    retrain resnet 56 0.1 40 0.8
    prune resnet 56 0.9
    retrain resnet 56 0.1 40 0.9
    prune resnet 56 0.95
    retrain resnet 56 0.1 40 0.95
}

random_resnet56(){
    random_prune resnet 56 0.2
    random_retrain resnet 56 0.1 40 0.2
    random_prune resnet 56 0.3
    random_retrain resnet 56 0.1 40 0.3
    random_prune resnet 56 0.4
    random_retrain resnet 56 0.1 40 0.4
    random_prune resnet 56 0.5
    random_retrain resnet 56 0.1 40 0.5
    random_prune resnet 56 0.6
    random_retrain resnet 56 0.1 40 0.6
    random_prune resnet 56 0.7
    random_retrain resnet 56 0.1 40 0.7
    random_prune resnet 56 0.8
    random_retrain resnet 56 0.1 40 0.8
    random_prune resnet 56 0.9
    random_retrain resnet 56 0.1 40 0.9
    random_prune resnet 56 0.95
    random_retrain resnet 56 0.1 40 0.95
}

resnet56 &&
random_resnet56 &&
resnet56 &&
random_resnet56 &&
resnet56 &&
random_resnet56