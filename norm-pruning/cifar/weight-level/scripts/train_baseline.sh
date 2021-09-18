python cifar.py --dataset cifar10 --arch vgg19_bn --depth 19 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar10 &&
python cifar.py --dataset cifar10 --arch preresnet --depth 110 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar10 &&
python cifar.py --dataset cifar10 --arch resnet --depth 110 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar10 &&
python cifar.py --dataset cifar10 --arch resnet --depth 56 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar10 &&
python cifar.py --dataset cifar10 --arch densenet --depth 40 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar10 &&
python cifar.py --dataset cifar10 --arch densenet --depth 100 --compressionRate 2 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar10 &&

python cifar.py --dataset cifar100 --arch vgg19_bn --depth 19 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar100 &&
python cifar.py --dataset cifar100 --arch preresnet --depth 110 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar100 &&
python cifar.py --dataset cifar100 --arch resnet --depth 110 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar100 &&
python cifar.py --dataset cifar100 --arch resnet --depth 56 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar100 &&
python cifar.py --dataset cifar100 --arch densenet --depth 40 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar100 &&
python cifar.py --dataset cifar100 --arch densenet --depth 100 --compressionRate 2 --save ../../../iclr2021_checkpoints/norm_pruning/weights/cifar/cifar100