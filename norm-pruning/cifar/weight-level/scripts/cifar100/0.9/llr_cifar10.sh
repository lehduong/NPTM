
prune(){
    python cifar_prune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar100 \
    --percent 0.9 \
    --resume ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar100/$1_$2.pt \
    --save_dir ../../../result/norm_pruning/weights/$1$2/llr
}

retrain(){
    python cifar_finetune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar100 \
    --resume ../../../result/norm_pruning/weights/$1$2/llr/pruned.pth.tar \
    --save_dir ../../../result/norm_pruning/weights/$1$2/llr/$4epochs \
    --lr $3 \
    --epochs $4 \
    --use_llr \
    --wandb_name $1_$2_90_llr_$4epochs
}

densenet40(){
    prune densenet 40
    retrain densenet 40 0.1 40
    retrain densenet 40 0.1 56
    retrain densenet 40 0.1 72
    retrain densenet 40 0.1 88
    retrain densenet 40 0.1 104
    retrain densenet 40 0.1 120
    retrain densenet 40 0.1 136
    retrain densenet 40 0.1 152
}

preresnet110(){
    prune preresnet 110
    retrain preresnet 110 0.1 40
    retrain preresnet 110 0.1 56
    retrain preresnet 110 0.1 72
    retrain preresnet 110 0.1 88
    retrain preresnet 110 0.1 104
    retrain preresnet 110 0.1 120
    retrain preresnet 110 0.1 136
    retrain preresnet 110 0.1 152
}

resnet110(){
    prune resnet 110
    retrain resnet 110 0.1 40
    retrain resnet 110 0.1 56
    retrain resnet 110 0.1 72
    retrain resnet 110 0.1 88
    retrain resnet 110 0.1 104
    retrain resnet 110 0.1 120
    retrain resnet 110 0.1 136
    retrain resnet 110 0.1 152
}

resnet56(){
    prune resnet 56
    retrain resnet 56 0.1 40
    retrain resnet 56 0.1 56
    retrain resnet 56 0.1 72
    retrain resnet 56 0.1 88
    retrain resnet 56 0.1 104
    retrain resnet 56 0.1 120
    retrain resnet 56 0.1 136
    retrain resnet 56 0.1 152
}

vgg19_bn(){
    prune vgg19_bn 19
    retrain vgg19_bn 19 0.1 40
    retrain vgg19_bn 19 0.1 56
    retrain vgg19_bn 19 0.1 72
    retrain vgg19_bn 19 0.1 88
    retrain vgg19_bn 19 0.1 104
    retrain vgg19_bn 19 0.1 120
    retrain vgg19_bn 19 0.1 136
    retrain vgg19_bn 19 0.1 152
}

resnet56
resnet110
preresnet110
vgg19_bn
densenet40