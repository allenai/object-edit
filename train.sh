#!/bin/bash

python main.py \
    -t \
    --base configs/sd-objaverse-$1.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --finetune_from zero123.ckpt \