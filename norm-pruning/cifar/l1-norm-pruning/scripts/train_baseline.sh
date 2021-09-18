python main.py --dataset cifar10 --arch vgg --depth 16 --save ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10 &&
python main.py --dataset cifar10 --arch resnet --depth 56 --save ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10 &&
python main.py --dataset cifar10 --arch resnet --depth 110 --save ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar10 &&
python main.py --dataset cifar100 --arch vgg --depth 16 --save ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100 &&
python main.py --dataset cifar100 --arch resnet --depth 56 --save ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100 &&
python main.py --dataset cifar100 --arch resnet --depth 110 --save ../../../iclr2021_checkpoints/norm_pruning/filters/cifar/cifar100