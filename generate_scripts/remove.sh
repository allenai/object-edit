#!/bin/bash

python run_generation.py \
    --task remove \
    --checkpoint_path remove.ckpt \
    --image_path demo_images/remove_cup.jpg \
    --object_prompt "the purple cup" \
    --device 1 \
    --cfg_scale 3.0