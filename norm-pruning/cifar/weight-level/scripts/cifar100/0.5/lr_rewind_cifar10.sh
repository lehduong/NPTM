
prune(){
    python cifar_prune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar100 \
    --percent 0.5 \
    --resume ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar100/$1_$2.pt \
    --save_dir ../../../result/norm_pruning/weights/$1$2/lr_rewind
}

retrain(){
    python cifar_finetune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar100 \
    --resume ../../../result/norm_pruning/weights/$1$2/lr_rewind/pruned.pth.tar \
    --save_dir ../../../result/norm_pruning/weights/$1$2/lr_rewind/$4epochs \
    --lr $3 \
    --epochs $4 \
    --use_lr_rewind \
    --wandb_name $1_$2_50_lr_rewind_$4epochs
}

densenet40(){
    prune densenet 40
    retrain densenet 40 0.001 40
    retrain densenet 40 0.001 56
    retrain densenet 40 0.001 72
    retrain densenet 40 0.001 88
    retrain densenet 40 0.001 104
    retrain densenet 40 0.001 120
    retrain densenet 40 0.001 136
    retrain densenet 40 0.001 152
}

preresnet110(){
    prune preresnet 110
    retrain preresnet 110 0.001 40
    retrain preresnet 110 0.001 56
    retrain preresnet 110 0.001 72
    retrain preresnet 110 0.001 88
    retrain preresnet 110 0.001 104
    retrain preresnet 110 0.001 120
    retrain preresnet 110 0.001 136
    retrain preresnet 110 0.001 152
}

resnet110(){
    prune resnet 110
    retrain resnet 110 0.001 40
    retrain resnet 110 0.001 56
    retrain resnet 110 0.001 72
    retrain resnet 110 0.001 88
    retrain resnet 110 0.001 104
    retrain resnet 110 0.001 120
    retrain resnet 110 0.001 136
    retrain resnet 110 0.001 152
}

resnet56(){
    prune resnet 56
    retrain resnet 56 0.001 40
    retrain resnet 56 0.001 56
    retrain resnet 56 0.001 72
    retrain resnet 56 0.001 88
    retrain resnet 56 0.001 104
    retrain resnet 56 0.001 120
    retrain resnet 56 0.001 136
    retrain resnet 56 0.001 152
}

vgg19_bn(){
    prune vgg19_bn 19
    retrain vgg19_bn 19 0.001 40
    retrain vgg19_bn 19 0.001 56
    retrain vgg19_bn 19 0.001 72
    retrain vgg19_bn 19 0.001 88
    retrain vgg19_bn 19 0.001 104
    retrain vgg19_bn 19 0.001 120
    retrain vgg19_bn 19 0.001 136
    retrain vgg19_bn 19 0.001 152
}

resnet56
resnet110
preresnet110
densenet40
vgg19_bn