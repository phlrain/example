export CUDA_VISIBLE_DEVICES=1
python3 train_pytorch.py --workers=15 --batch_size=128 --data_dir=./ILSVRC2012_Pytorch/dataset_100/  --model=ResNet50
