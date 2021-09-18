python main.py --name=runs/resnet50/resnet50_prune56 --dataset=Imagenet \
--lr=0.1 --lr-decay-every=10 --momentum=0.9 --epochs=120 --batch-size=256 \
--pruning=True --seed=0 --model=resnet50 --load_model=./models/pretrained/resnet50-19c8e357.pth \
--mgpu=True --group_wd_coeff=1e-8 --wd=0.0 --tensorboard=True --pruning-method=22 \
--data=../data/image-net/ --no_grad_clip=True --pruning_config=./configs/imagenet_resnet50_prune56.json \
--use_onecycle --wandb_name resnet50_56_onecycle