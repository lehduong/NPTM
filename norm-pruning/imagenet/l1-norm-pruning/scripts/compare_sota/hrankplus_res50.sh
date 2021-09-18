python residual_prune.py --save checkpoints/hrankplus1105 \
--arch resnet50 \
--data ../../../data/image-net \
-v HRankPlus11.05 &&

python main_finetune.py \
--arch resnet50 \
--refine checkpoints/hrankplus1105/pruned.pth.tar \
--save checkpoints/hrankplus1105 \
--data ../../../data/image-net \
--epochs 90 \
--use_onecycle \
--lr 0.1 \
--wandb resnet_50_HRankPlus11.05_onecycle_0.1