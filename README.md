# OBJect 3DIT: Language-guided 3D-aware Image Editing

This is the official codebase for the paper [OBJect 3DIT: Language-guided 3D-aware Image Editing](https://arxiv.org/abs/2307.11073).

## Set-up

First clone this repository.

```bash
git clone https://github.com/allenai/object-edit.git
cd object-edit
mkdir DATASET
```

Optionally, create a conda environment.

```bash
conda create --name object python=3.9
conda activate object
```
Install all of the requirements.
```bash
. setup_reqs.sh
```
Download the dataset from HuggingFace [here](https://huggingface.co/datasets/allenai/object-edit/tree/main). By default the dataset is expected to be in the `DATASET` dir, so download them here. You can also change the default path of the dataset in the config files.
Unzip with the following commands
```
cd DATASET
tar -xzvf TASK.tar.gz
rm TASK.tar.gz
```
where `TASK` is either "remove", "rotate" or "translate". There is no extra data needed for the insertion task since it uses the same data as removal.

The dataset has the following structure.
```
DATASET/
└── rotate/
    └── train/
        ├── uid/
        │   ├── 1.png
        │   ├── 2.png
        │   ├── 1_mask.png
        |   ├── 2_mask.png
        │   └── metadata.json
```
The checkpoints for trained editing models and Zero123 initialization can also be found on the Huggingface page for this project.
## Training

If you would like to finetune from a [Zero123](https://github.com/cvlab-columbia/zero123) or [Image-Conditioned StableDiffusion](https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned) checkpoint, please download and modify the path in `train.sh`. If you would like to train from scratch, then delete the `--finetune_from` argument from `train.sh`. You may also change the devices used in the `--gpus` argument. To train, run the following, replacing `TASK` with either "rotate","remove","insert","translate" or "multitask":
```bash
. train.sh TASK
```

## Inference demo
You can run the scripts in `generate_scripts` to see inference in each of the editing tasks.
```
. generate_scripts/rotate.sh
. generate_scripts/remove.sh
. generate_scripts/insert.sh
. generate_scripts/translate.sh
```
They each run the `run_generation.py` script. You can modify the arguments in these shell scripts to perform different edits. Note that the object prompt should not contain the editing instruction, that will be filled in automatically. You only need to put in a description of the targeted object.
## Evaluation

If you would like to evaluate your generated images on the benchmark, you can run:

```
python run_eval.py \
--generation_dir YOUR_GENERATED_IMAGES_PATH \
--data_dir PATH_OF_OBJECT_DATASET \
--task [rotate|remove|insert|translate] \
--split [train|val|test] \
--seen [seen|unseen] \
--save_dir PATH_TO_SAVE_STATISTICS_SUMMARY
```

This script assumes your generated images are saved in the directory with the same UID as the corresponding sample in the dataset.
```
base_path/
│
├── uid/
│   ├── 0.png
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
