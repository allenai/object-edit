#!/bin/bash
pip install -r req.txt
pip install pytorch_lightning==1.5.0
pip install omegaconf
pip install opencv-python
pip install carvekit-colab==4.1.0
pip install einops
pip install taming-transformers-rom1504 
pip install kornia
pip install git+https://github.com/openai/CLIP.git
pip install transformers
pip install wandb
pip install webdataset==0.2.5
pip install ipdb
pip install matplotlib
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install torchmetrics[image]