export CUDA_VISIBLE_DEVICES=0

DATA_PATH=./ILSVRC2012_Pytorch/dataset_100/

MODEL_NAME=ResNet50
BATCH_SIZE=128
is_paddle=1
is_fake=0

while getopts m:b:p:f: opt
do  
    case $opt in
        m)
            MODEL_NAME="$OPTARG"
            ;;
        b)
            BATCH_SIZE="$OPTARG"
            ;;
        p)
            is_paddle="$OPTARG"
            ;;
        f)
            is_fake="$OPTARG"
            ;;
        \?)
            exit;  
            ;;
    esac
done

# echo $DATA_PATH $MODEL_NAME $BATCH_SIZE $is_paddle $is_fake
script=train_paddle.py

if [ $is_paddle -eq 1 ]; then
  if [ $is_fake -eq 1 ]; then
  script=fake_paddle.py
  else
  script=train_paddle.py
  fi
else
  if [ $is_fake -eq 1 ]; then
  script=fake_pytorch.py
  else
  script=train_pytorch.py
  fi
fi

echo $script

python3 $script --batch_size=$BATCH_SIZE --data_dir=$DATA_PATH --model=$MODEL_NAME

