python main.py --dataset cifar10 --arch vgg --depth 16 --save vgg16/standard/seed_1 --seed 1 --wandb_name vgg16_standard &&
python main.py --dataset cifar10 --arch vgg --depth 16 --save vgg16/onecycle/seed_1 --seed 1 --use_onecycle --wandb_name vgg16_onecycle &&
python main.py --dataset cifar10 --arch resnet --depth 56 --save resnet_56/standard/seed_1 --seed 1 --wandb_name resnet56_standard &&
python main.py --dataset cifar10 --arch resnet --depth 56 --save resnet_56/onecycle/seed 1 --seed 1 --use_onecycle --wandb_name resnet56_onecycle &&
python main.py --dataset cifar10 --arch resnet --depth 110 --save resnet_110/standard/seed_1 --seed 1 --wandb_name resnet110_standard &&
python main.py --dataset cifar10 --arch resnet --depth 110 --save resnet_110/onecycle/seed_1 --seed 1 --use_onecycle --wandb_name resnet110_onecycle &&

python main.py --dataset cifar10 --arch vgg --depth 16 --save vgg16/standard/seed_2 --seed 2 --wandb_name vgg16_standard &&
python main.py --dataset cifar10 --arch vgg --depth 16 --save vgg16/onecycle/seed_2 --seed 2 --use_onecycle --wandb_name vgg16_onecycle &&
python main.py --dataset cifar10 --arch resnet --depth 56 --save resnet_56/standard/seed_2 --seed 2 --wandb_name resnet56_standard &&
python main.py --dataset cifar10 --arch resnet --depth 56 --save resnet_56/onecycle/seed 2 --seed 2 --use_onecycle --wandb_name resnet56_onecycle &&
python main.py --dataset cifar10 --arch resnet --depth 110 --save resnet_110/standard/seed_2 --seed 2 --wandb_name resnet110_standard &&
python main.py --dataset cifar10 --arch resnet --depth 110 --save resnet_110/onecycle/seed_2 --seed 2 --use_onecycle --wandb_name resnet110_onecycle &&

python main.py --dataset cifar10 --arch vgg --depth 16 --save vgg16/standard/seed_3 --seed 3 --wandb_name vgg16_standard &&
python main.py --dataset cifar10 --arch vgg --depth 16 --save vgg16/onecycle/seed_3 --seed 3 --use_onecycle --wandb_name vgg16_onecycle &&
python main.py --dataset cifar10 --arch resnet --depth 56 --save resnet_56/standard/seed_3 --seed 3 --wandb_name resnet56_standard &&
python main.py --dataset cifar10 --arch resnet --depth 56 --save resnet_56/onecycle/seed 3 --seed 3 --use_onecycle --wandb_name resnet56_onecycle &&
python main.py --dataset cifar10 --arch resnet --depth 110 --save resnet_110/standard/seed_3 --seed 3 --wandb_name resnet110_standard &&
python main.py --dataset cifar10 --arch resnet --depth 110 --save resnet_110/onecycle/seed_3 --seed 3 --use_onecycle --wandb_name resnet110_onecycle