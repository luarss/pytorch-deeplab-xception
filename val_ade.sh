CUDA_VISIBLE_DEVICES=0 /home/luarss/anaconda3/bin/python validation.py --backbone resnet --lr 0.007 --workers 4 --epochs 40 --batch-size 5 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset ade --resume ./run/ade/deeplab-resnet/experiment_41/checkpoint.pth.tar

