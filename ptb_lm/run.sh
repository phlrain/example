
export CUDA_VISIBLE_DEVICES=2

python main.py --cuda --emsize 200 --nhid 200 --dropout 0.0 --epochs 1 --bptt 20 --data data/simple-examples/data/
#python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 1 --bptt 35 --data data/simple-examples/data/
#python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.5 --epochs 1 --bptt 35 --data data/simple-examples/data/
