#!/bin/bash

# This yields a test perplexity of 63.358.
# perfgormance can be increased further by training for longer and increasing the
# dropoute and dropouti parameters.
# It seems that dropouto is undertunes. Performance might be improved by increasing it
# fruther.
python train.py --cuda --fp16 --data ../data/wikitext-2/ --dataset wt103 --adaptive --gpu0_bsz 1 --dynamic-loss-scale --log-interval 50 \
--eval-interval 400 --d_model 400 --n_head 10 --d_head 40 --warmup_step 3000 --tgt_len 150 --mem_len 150 --eval_tgt_len 150 --batch_size 32 \
--seed 0   --lr 0.00035  --max_step 125000  --dropouti 0.7  --dropouto 0.5  --dropoute 0.3  --dropout 0.2  --dropatt 0.2  --d_inner 900  --n_layer 16 --seed 11
