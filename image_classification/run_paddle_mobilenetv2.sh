export CUDA_VISIBLE_DEVICES=0
python3 train_paddle.py --batch_size=128 --data_dir=./ILSVRC2012_Pytorch/dataset_100/  --model=MobileNetV2
