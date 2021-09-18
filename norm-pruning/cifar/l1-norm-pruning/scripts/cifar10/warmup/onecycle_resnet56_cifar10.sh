prune_warmup_ablation(){
    python res${4}prune.py \
    --dataset cifar10 \
    -v $1 \
    --model ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10/resnet_$4.pt \
    --save cifar10/resnet$4/ &&
    
    python main_finetune.py \
    --refine cifar10/resnet$4/pruned.pth.tar \
    --save ../../../result/norm_pruning/filters/resnet$4/onecycle/${3}epochs \
    --dataset cifar10 \
    --arch resnet \
    --pct_start $2 \
    --depth $4 \
    --epochs $3 \
    --use_onecycle \
    --lr 0.1 \
    --wandb_name resnet_$4_$1_onecycle_${3}epochs_${2}warmup
}

run(){
    prune_warmup_ablation B $1 40 56
    prune_warmup_ablation B $1 56 56
    prune_warmup_ablation B $1 72 56
    prune_warmup_ablation B $1 88 56
    prune_warmup_ablation B $1 104 56
    prune_warmup_ablation B $1 120 56
    prune_warmup_ablation B $1 136 56
    prune_warmup_ablation B $1 152 56
}

run 0
run 0.05
run 0.15