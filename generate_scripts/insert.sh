#!/bin/bash

python run_generation.py \
    --task insert \
    --checkpoint_path insert.ckpt \
    --image_path demo_images/insert_yard.png \
    --object_prompt "a blue bookcase" \
    --position 0.25,0.6 \
    --device 1 \
    --cfg_scale 3.0