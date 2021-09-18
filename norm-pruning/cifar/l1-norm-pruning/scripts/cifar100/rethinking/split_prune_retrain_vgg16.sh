prune_retrain_vgg16(){
    # train
    python train_test_split_main.py --dataset cifar100 \
    --arch vgg \
    --depth 16 \
    --save prune_retrain_checkpoints/vgg16_$2 \
    --seed $2 \
    --wandb vgg_16_standard_train_test_split &&
    
    # prune A
    python vggprune.py \
    --dataset cifar100 \
    -v A \
    --model prune_retrain_checkpoints/vgg16_$2/vgg_16_best.pt \
    --save prune_retrain_checkpoints/vgg16_$2 &&
    
    # onecycle
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/vgg16_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/vgg16_$2 \
    --dataset cifar100 \
    --arch vgg \
    --depth 16 \
    --epochs 40 \
    --use_onecycle \
    --seed $2 \
    --lr 0.1 \
    --wandb_name vgg_16_A_onecycle_40epochs_train_test_split &&
    
    # fine-tune
    python train_test_split_finetune.py \
    --refine prune_retrain_checkpoints/vgg16_$2/pruned.pth.tar \
    --save prune_retrain_checkpoints/vgg16_$2 \
    --dataset cifar100 \
    --arch vgg \
    --depth 16 \
    --epochs 40 \
    --seed $2 \
    --wandb_name vgg_16_A_finetune_40epochs_train_test_split
    
}

prune_retrain_vgg16 B 1 &&
prune_retrain_vgg16 B 2 &&
prune_retrain_vgg16 B 3 &&
prune_retrain_vgg16 B 4 &&
prune_retrain_vgg16 B 5