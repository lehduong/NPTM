train_baseline_clr(){
    python main.py --dataset cifar10 \
    --arch resnet \
    --depth 56 \
    --save resnet_56/onecycle/seed_$1/baseline \
    --seed $1 \
    --epochs 160 \
    --use_onecycle \
    --wandb_name resnet56_baseline_clr
}

prune(){
    python res56prune.py \
    --dataset cifar10 \
    -v $2 \
    --model resnet_56/onecycle/seed_$1/baseline/resnet_56_best.pt \
    --save resnet_56/onecycle/seed_$1/pruned
}

finetune(){
    python main_finetune.py \
    --refine resnet_56/onecycle/seed_$1/pruned/pruned.pth.tar \
    --save resnet_56/onecycle/seed_$1/finetune \
    --dataset cifar10 \
    --arch resnet \
    --depth 56 \
    --seed $1 \
    --epochs $2 \
    --wandb_name resnet_56_clr_$3_finetune_${2}epochs
}

clr_finetune(){
    python main_finetune.py \
    --refine resnet_56/onecycle/seed_$1/pruned/pruned.pth.tar \
    --save resnet_56/onecycle/seed_$1/clr \
    --dataset cifar10 \
    --arch resnet \
    --depth 56 \
    --epochs $2 \
    --seed $1 \
    --use_onecycle \
    --lr 0.1 \
    --wandb_name resnet_56_clr_$3_clr_${2}epochs
}

slr_finetune(){
    python main_finetune.py \
    --refine resnet_56/onecycle/seed_$1/pruned/pruned.pth.tar \
    --save resnet_56/onecycle/seed_$1/slr \
    --dataset cifar10 \
    --arch resnet \
    --depth 56 \
    --epochs $2 \
    --seed $1 \
    --use_llr \
    --lr 0.1 \
    --wandb_name resnet_56_clr_$3_slr_${2}epochs
}

run(){
    train_baseline_clr $1 $2
    prune $1 $3
    finetune $1 $2 $3
    clr_finetune $1 $2 $3
    slr_finetune $1 $2 $3
}

run 1 40 B
run 2 40 B
run 3 40 B