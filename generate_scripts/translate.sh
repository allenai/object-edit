#!/bin/bash

python run_generation.py \
    --task translate \
    --checkpoint_path translate.ckpt \
    --image_path demo_images/move_cube.jpg \
    --object_prompt "the blue cube" \
    --position 0.8,0.2 \
    --device 1 \
    --cfg_scale 3.0