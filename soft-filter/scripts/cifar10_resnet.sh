#!/bin/bash


change_layer_end_for_different_structure(){
    resnet110 324
    resnet56 162
    resnet32 90
    resnet20 54
}

standard_training(){
    depth=$1
    layerEnd=$2
    finetuneEpochs=$3
    seed=$4
    keptRate=$5
    
    python ablation_pruning_cifar10_resnet.py  ../data/ --dataset cifar10 --arch resnet${depth} \
    --save_path ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_baseline \
    --epochs 200 \
    --finetune_epochs 0 \
    --schedule 1 60 120 160 \
    --gammas 10 0.2 0.2 0.2 \
    --learning_rate 0.01 --decay 0.0005 --batch_size 128 \
    --rate $keptRate \
    --manualSeed $seed \
    --layer_begin 0  --layer_end $layerEnd --layer_inter 3 --epoch_prune 1 \
    --wandb_name resnet_${depth}_${keptRate}_baseline_200_epochs
}

standard_retraining(){
    depth=$1
    layerEnd=$2
    finetuneEpochs=$3
    seed=$4
    keptRate=$5
    
    # make parent dir to store multiple checkpoints with different epochs
    if [ ! -d "./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_standard" ]
    then
        mkdir ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_standard
    fi
    
    # make child dir to store checkpoints with specified epochs
    if [ ! -d "./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_standard/${finetuneEpochs}epochs" ]
    then
        mkdir ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_standard/${finetuneEpochs}epochs
    fi
    
    # cp best val model to current folder
    cp ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_baseline/model_best.pth.tar \
    ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_standard/${finetuneEpochs}epochs &&
    
    # run
    python ablation_pruning_cifar10_resnet.py  ../data/ --dataset cifar10 --arch resnet${depth} \
    --save_path ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_standard/${finetuneEpochs}epochs \
    --epochs 200 \
    --finetune_epochs $finetuneEpochs \
    --schedule 1 60 120 160 \
    --gammas 10 0.2 0.2 0.2 \
    --learning_rate 0.01 --decay 0.0005 --batch_size 128 \
    --rate $keptRate \
    --resume ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_standard/${finetuneEpochs}epochs/model_best.pth.tar \
    --finetune \
    --manualSeed $seed \
    --layer_begin 0  --layer_end $layerEnd --layer_inter 3 --epoch_prune 1 \
    --wandb_name resnet_${depth}_${keptRate}_standard_${finetuneEpochs}epochs
}

onecycle_retraining(){
    depth=$1
    layerEnd=$2
    finetuneEpochs=$3
    seed=$4
    keptRate=$5
    
    # make parent dir to store multiple checkpoints with different epochs
    if [ ! -d "./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_onecycle" ]
    then
        mkdir ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_onecycle
    fi
    
    # make child dir to store checkpoints with specified epochs
    if [ ! -d "./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_onecycle/${finetuneEpochs}epochs" ]
    then
        mkdir ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_onecycle/${finetuneEpochs}epochs
    fi
    
    # cp best val model to current folder
    cp ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_baseline/model_best.pth.tar \
    ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_onecycle/${finetuneEpochs}epochs &&
    
    # run
    python ablation_pruning_cifar10_resnet.py  ../data/ --dataset cifar10 --arch resnet${depth} \
    --save_path ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_onecycle/${finetuneEpochs}epochs \
    --epochs 200 \
    --finetune_epochs $finetuneEpochs \
    --schedule 1 60 120 160 \
    --gammas 10 0.2 0.2 0.2 \
    --learning_rate 0.01 --decay 0.0005 --batch_size 128 \
    --rate $keptRate \
    --resume ./checkpoints/cifar10/cifar10_resnet${depth}_rate${keptRate}_${seed}_onecycle/${finetuneEpochs}epochs/model_best.pth.tar \
    --manualSeed $seed \
    --layer_begin 0  --layer_end $layerEnd --layer_inter 3 --epoch_prune 1 \
    --use_onecycle \
    --finetune \
    --wandb_name resnet_${depth}_${keptRate}_onecycle_${finetuneEpochs}epochs
}

run_exp(){
    standard_training 56 162 0 $1 $2 # depth, layer_end, finetune_epochs, seed, rate
    onecycle_retraining 56 162 50 $1 $2
    onecycle_retraining 56 162 100 $1 $2
    onecycle_retraining 56 162 150 $1 $2
    onecycle_retraining 56 162 200 $1 $2
    standard_retraining 56 162 200 $1 $2
    
    standard_training 110 324 0 $1 $2 # depth, layer_end, finetune_epochs, seed, rate
    onecycle_retraining 110 324 50 $1 $2
    onecycle_retraining 110 324 100 $1 $2
    onecycle_retraining 110 324 150 $1 $2
    onecycle_retraining 110 324 200 $1 $2
    standard_retraining 110 324 200 $1 $2
}

run_exp 1 0.7
run_exp 2 0.7
run_exp 3 0.7