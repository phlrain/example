CUDA_VISIBLE_DEVICES=1 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch lstm_luong_wmt_en_de  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion cross_entropy \
    --log-format 'simple' --log-interval 100 \
    --use-basic \
    --max-sentences 128 \
    --decoder-attention False
