initial_prune(){python cifar_prune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar10 \
    --percent $3 \
    --resume ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar10/$1_$2.pt \
    --save_dir ../../../result/norm_pruning/weights/$1$2/onecycle/prune_$4
}

later_prune(){
    let previous=$4-1
    python cifar_prune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar10 \
    --percent $3 \
    --resume ../../../result/norm_pruning/weights/$1$2/onecycle/prune_$previous/finetuned.pth.tar \
    --save_dir ../../../result/norm_pruning/weights/$1$2/onecycle/prune_$4
}

finetune(){
    python cifar_finetune.py \
    --arch $1 \
    --depth $2 \
    --dataset cifar10 \
    --resume ../../../result/norm_pruning/weights/$1$2/onecycle/prune_$4/pruned.pth.tar \
    --save_dir ../../../result/norm_pruning/weights/$1$2/onecycle/prune_$4 \
    --lr 0.1 \
    --use_onecycle \
    --wandb_name iterative_preresnet_110_$3_onecycle
}

resnet56(){
    # first prune, 30%
    initial_prune resnet 56 0.3 1
    finetune resnet 56 30 1
    # second prune, 51%
    later_prune resnet 56 0.51 2
    finetune resnet 56 51 2
    # third prune, 66%
    later_prune resnet 56 0.66 3
    finetune resnet 56 66 3
    # fourth prune, 76%
    later_prune resnet 56 0.76 4
    finetune resnet 56 76 4
    # fifth prune, 83%
    later_prune resnet 56 0.83 5
    finetune resnet 56 83 5
    # sixth prune, 88%
    later_prune resnet 56 0.88 6
    finetune resnet 56 88 6
    # seventh prune, 92%
    later_prune resnet 56 0.92 7
    finetune resnet 56 92 7
    # eigth prune, 94%
    later_prune resnet 56 0.94 8
    finetune resnet 56 94 8
    # ninth prune, 96%
    later_prune resnet 56 0.96 9
    finetune resnet 56 96 9
}


resnet110(){
    # first prune, 30%
    initial_prune resnet 110 0.3 1
    finetune resnet 110 30 1
    # second prune, 51%
    later_prune resnet 110 0.51 2
    finetune resnet 110 51 2
    # third prune, 66%
    later_prune resnet 110 0.66 3
    finetune resnet 110 66 3
    # fourth prune, 76%
    later_prune resnet 110 0.76 4
    finetune resnet 110 76 4
    # fifth prune, 83%
    later_prune resnet 110 0.83 5
    finetune resnet 110 83 5
    # sixth prune, 88%
    later_prune resnet 110 0.88 6
    finetune resnet 110 88 6
    # seventh prune, 92%
    later_prune resnet 110 0.92 7
    finetune resnet 110 92 7
    # eigth prune, 94%
    later_prune resnet 110 0.94 8
    finetune resnet 110 94 8
    # ninth prune, 96%
    later_prune resnet 110 0.96 9
    finetune resnet 110 96 9
}

resnet56
resnet110