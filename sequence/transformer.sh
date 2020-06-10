mkdir -p checkpoints/fconv_wmt_en_de
CUDA_VISIBLE_DEVICES=2 fairseq-train \
    data-bin/wmt17_en_de \
    --arch transformer_wmt_en_de \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/fconv_wmt_en_de
