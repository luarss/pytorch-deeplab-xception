CUDA_VISIBLE_DEVICES=0 /home/luarss/anaconda3/bin/python transfer.py --backbone resnet --lr 0.007 --workers 4 --epochs 1 --batch-size 5 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset ade --resume ./deeplab-resnet.pth.tar --no-val 

#./run/ade/deeplab-resnet/experiment_5/checkpoint.pth.tar

# resume ./deeplab-resnet.pth.tar
