
prune(){
    python cifar_prune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar100 \
    --percent 0.8 \
    --resume ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar100/$1_$2.pt \
    --save_dir ../../../result/norm_pruning/weights/$1$2/standard
}

retrain(){
    python cifar_finetune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar100 \
    --resume ../../../result/norm_pruning/weights/$1$2/standard/pruned.pth.tar \
    --save_dir ../../../result/norm_pruning/weights/$1$2/standard/$4epochs \
    --lr $3 \
    --epochs $4 \
    --wandb_name $1_$2_80_standard_$4epochs
}

densenet40(){
    prune densenet 40
    retrain densenet 40 0.001 300
}

preresnet110(){
    prune preresnet 110
    retrain preresnet 110 0.001 300
}

resnet110(){
    prune resnet 110
    retrain resnet 110 0.001 300
}

resnet56(){
    prune resnet 56
    retrain resnet 56 0.001 300
}

vgg19_bn(){
    prune vgg19_bn 19
    retrain vgg19_bn 19 0.001 300
}

resnet56
resnet110
preresnet110
densenet40
vgg19_bn