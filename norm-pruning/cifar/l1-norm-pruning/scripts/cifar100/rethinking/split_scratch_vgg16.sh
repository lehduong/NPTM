scratch_vgg16(){
    python train_test_split_main.py --dataset cifar100 \
    --arch vgg \
    --depth 16 \
    --save scratch_checkpoints/vgg16_$2 \
    --epochs 1 \
    --wandb_name redundant &&
    
    python vggprune.py \
    --dataset cifar100 \
    -v $1 \
    --model scratch_checkpoints/vgg16_$2/vgg_16_best.pt \
    --save scratch_checkpoints/vgg16_$2 &&
    
    python train_test_split_main_B.py \
    --scratch scratch_checkpoints/vgg16_$2/pruned.pth.tar \
    --save scratch_checkpoints/vgg16_$2 \
    --dataset cifar100 \
    --arch vgg \
    --depth 16 \
    --seed $2 \
    --wandb_name vgg_16_${1}_scratch_train_test_split
}

scratch_vgg16 A 1 &&
scratch_vgg16 A 2 &&
scratch_vgg16 A 3 &&
scratch_vgg16 A 4 &&
scratch_vgg16 A 5