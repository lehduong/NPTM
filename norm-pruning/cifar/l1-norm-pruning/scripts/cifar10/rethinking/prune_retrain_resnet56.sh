prune_retrain_resnet56(){
    python main.py --dataset cifar10 --arch resnet --depth 56 --save main_checkpoints/resnet56_$2 &&
    
    python res56prune.py \
    --dataset cifar10 \
    -v $1 \
    --model main_checkpoints/resnet56_$2/resnet_56_best.pt \
    --save prune_retrain_checkpoints/resnet56_$2 &&
    
    python main_finetune.py \
    --refine prune_retrain_checkpoints/resnet56_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/resnet56_$2 \
    --dataset cifar10 \
    --arch resnet \
    --depth 56 \
    --epochs 40 \
    --use_onecycle \
    --seed $2 \
    --lr 0.1 \
    --wandb_name resnet_56_${1}_onecycle_40epochs
}

prune_retrain_resnet56 B 1 &&
prune_retrain_resnet56 B 2 &&
prune_retrain_resnet56 B 3 &&
prune_retrain_resnet56 A 1 &&
prune_retrain_resnet56 A 2 &&
prune_retrain_resnet56 A 3