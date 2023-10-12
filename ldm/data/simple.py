from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
# from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
import clip
import zipfile
from torch.utils.data.distributed import DistributedSampler


class ObjaverseTaskDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, task,batch_size, train=None, validation=None,
                 test=None, num_workers=4,num_samples=100000, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples

        self.task_classes = {
            "rotate": ObjaverseDataRotation,
            "translate": ObjaverseDataTranslation,
            "remove": ObjaverseDataRemove,
            "insert": ObjaverseDataInsert,
            "multitask": ObjaverseDataMultitask

        }

        assert task in self.task_classes

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = self.task_classes[self.task](
                root_dir=self.root_dir, 
                image_transforms=self.image_transforms,
                task=self.task,
                split="train",
                seen_or_unseen="seen",
                num_samples=self.num_samples
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset_seen = self.task_classes[self.task](
            root_dir=self.root_dir, 
            image_transforms=self.image_transforms,
            task=self.task,
            split="val",
            seen_or_unseen="seen",
            num_samples=512
        )
        loader_seen = wds.WebLoader(dataset_seen, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        dataset_unseen = self.task_classes[self.task](
            root_dir=self.root_dir, 
            image_transforms=self.image_transforms,
            task=self.task,
            split="val",
            seen_or_unseen="unseen",
            num_samples=512
        )
        loader_unseen = wds.WebLoader(dataset_unseen, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return [loader_seen,loader_unseen]


    def test_dataloader(self):
        dataset_seen = self.task_classes[self.task](
            root_dir=self.root_dir, 
            image_transforms=self.image_transforms,
            task=self.task,
            split="test",
            seen_or_unseen="seen",
            num_samples=self.num_samples
        )
        loader_seen = wds.WebLoader(dataset_seen, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        dataset_unseen = self.task_classes[self.task](
            root_dir=self.root_dir, 
            image_transforms=self.image_transforms,
            task=self.task,
            split="test",
            seen_or_unseen="unseen",
            num_samples=self.num_samples
        )
        loader_unseen = wds.WebLoader(dataset_unseen, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return [loader_seen,loader_unseen]

class ObjaverseDataRotation(Dataset):
    def __init__(self,
        root_dir='.DATASET',
        image_transforms=[],
        task="rotate",
        split="train",
        seen_or_unseen="seen",
        num_samples=100000
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """

        assert not (split == "train" and seen_or_unseen == "unseen")
        assert task == "rotate"

        self.split = split

        if split == "train":
            task_split_seen = os.path.join(task,split)
        else:
            task_split_seen = os.path.join(task,split,seen_or_unseen)

        self.root_dir = os.path.join(root_dir,task_split_seen)
        self.samples = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir,f))][:num_samples]
        self.len = len(self.samples)
        self.log_key = task_split_seen

        with open("objaverse_cat_descriptions_64k.json","r") as f:
            self.object_annot = json.load(f)

        
        self.prompt_templates = ["rotate the {}"]

        print('============= length of dataset %d =============' % self.len)

        self.tform = image_transforms

    def __len__(self):

        return self.len
        
        
    def get_T(self, rotation):
        azimuth = math.radians(rotation)
        d_T = torch.tensor([np.pi / 2, math.sin(azimuth), math.cos(azimuth),0]) # match format of Zero123
        return d_T

    def __getitem__(self, index):

        data = {}

        try:
            sample_dir = self.samples[index]
            cond_im, target_im, mask_cond, mask_target, metadata = get_sample(self.root_dir,sample_dir,self.tform)
        except Exception as e: 
            sample_dir = self.samples[0]
            cond_im, target_im, mask_cond, mask_target, metadata = get_sample(self.root_dir,sample_dir,self.tform)
        
        rotation_angle = metadata["rotation_angle"]
        
        assert metadata["rotation_category"] == metadata["object_data"][0]["category"]
        prompt = self.object_annot[metadata["object_data"][0]["uid"]]["description"]
        prompt = "rotate the {}".format(prompt)


        data["mask_cond"] = mask_cond
        data["mask_target"] = mask_target 
        data["uid"] = sample_dir
        data["image_target"] = target_im
        data["cond"] = {
            "image": cond_im,
            "text": prompt
        }
        data["T"] = self.get_T(rotation_angle)
        data["log_key"] = self.log_key
        
        return data

class ObjaverseDataTranslation(Dataset):
    def __init__(self,
        root_dir='.DATASET',
        image_transforms=[],
        task="translate",
        split="train",
        seen_or_unseen="seen",
        num_samples=100000
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        
        assert not (split == "train" and seen_or_unseen == "unseen")
        assert task == "translate"

        self.split = split

        if split == "train":
            task_split_seen = os.path.join(task,split)
        else:
            task_split_seen = os.path.join(task,split,seen_or_unseen)

        with open("objaverse_cat_descriptions_64k.json","r") as f:
            self.object_annot = json.load(f)

        self.root_dir = os.path.join(root_dir,task_split_seen)
        self.samples = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir,f))][:num_samples]
        self.len = len(self.samples)
        self.log_key = task_split_seen
        
        self.prompt_templates = ["move the {}"]

        print('============= length of dataset %d =============' % self.len)

        self.tform = image_transforms

    def __len__(self):

        return self.len
        
        
    def get_T(self, x,y):
        d_T = torch.tensor([0,x,y,0])
        return d_T

    def __getitem__(self, index):

        data = {}
        try:
            sample_dir = self.samples[index]
            cond_im, target_im, mask_cond, mask_target, metadata = get_sample(self.root_dir,sample_dir,self.tform)
        except Exception as e: 
            sample_dir = self.samples[0]
            cond_im, target_im, mask_cond, mask_target, metadata = get_sample(self.root_dir,sample_dir,self.tform)
        
        translation = metadata["end_location_2d"]
        
        assert metadata["translation_category"] == metadata["object_data"][0]["category"]
        prompt = self.object_annot[metadata["object_data"][0]["uid"]]["description"]
        prompt = "move the {}".format(prompt)


        data["mask_cond"] = mask_cond
        data["mask_target"] = mask_target 
        data["uid"] = sample_dir
        data["image_target"] = target_im
        data["cond"] = {
            "image": cond_im,
            "text": prompt
        }
        data["T"] = self.get_T(translation[0],translation[1])
        data["log_key"] = self.log_key

        return data

class ObjaverseDataRemove(Dataset):
    def __init__(self,
        root_dir='.DATASET',
        image_transforms=[],
        task="remove",
        split="train",
        seen_or_unseen="seen",
        num_samples=100000
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        
        assert not (split == "train" and seen_or_unseen == "unseen")
        assert task == "remove"

        self.split = split

        if split == "train":
            task_split_seen = os.path.join(task,split)
        else:
            task_split_seen = os.path.join(task,split,seen_or_unseen)

        self.root_dir = os.path.join(root_dir,task_split_seen)
        self.samples = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir,f))][:num_samples]
        self.len = len(self.samples)
        self.log_key = task_split_seen

        with open("objaverse_cat_descriptions_64k.json","r") as f:
            self.object_annot = json.load(f)

        self.prompt_templates = ["delete the {}"]

        print('============= length of dataset %d =============' % self.len)

        self.tform = image_transforms

    def __len__(self):

        return self.len
        
        
    def get_T(self):
        d_T = torch.tensor([0.,0.,0.,0.])
        return d_T

    def __getitem__(self, index):

        data = {}

        try:
            sample_dir = self.samples[index]
            cond_im, target_im, mask_cond, mask_target, metadata = get_sample(self.root_dir,sample_dir,self.tform)
        except Exception as e: 
            sample_dir = self.samples[0]
            cond_im, target_im, mask_cond, mask_target, metadata = get_sample(self.root_dir,sample_dir,self.tform)
                
        prompt = self.object_annot[metadata["removed_object_uid"]]["description"]
        prompt = "remove the {}".format(prompt)
        data["mask_cond"] = mask_cond
        data["mask_target"] = mask_cond # dont use mask_target because it has no object (it was removed)
        data["uid"] = sample_dir
        data["image_target"] = target_im
        data["cond"] = {
            "image": cond_im,
            "text": prompt
        }
        data["T"] = self.get_T()
        data["log_key"] = self.log_key
       
        return data

class ObjaverseDataInsert(Dataset):
    def __init__(self,
        root_dir='.DATASET',
        image_transforms=[],
        task="insert",
        split="train",
        seen_or_unseen="seen",
        num_samples=100000
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        
        assert not (split == "train" and seen_or_unseen == "unseen")
        assert task == "insert"

        self.split = split

        if split == "train":
            split_seen = os.path.join(split)
        else:
            split_seen = os.path.join(split,seen_or_unseen)

        self.root_dir = os.path.join(root_dir,os.path.join("remove",split_seen)) # we just use remove data and invert it
        self.samples = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir,f))][:num_samples]
        self.len = len(self.samples)
        # assert self.len == num_samples
        self.log_key = os.path.join(task,split_seen)
        
        with open("objaverse_cat_descriptions_64k.json","r") as f:
            self.object_annot = json.load(f)

        print('============= length of dataset %d =============' % self.len)

        self.tform = image_transforms

    def __len__(self):

        return self.len
        
        
    def get_T(self, x,y):
        d_T = torch.tensor([0,x,y,0])
        return d_T

    def __getitem__(self, index):

        data = {}

        try:
            sample_dir = self.samples[index]
            target_im, cond_im, mask_target, mask_cond, metadata = get_sample(self.root_dir,sample_dir,self.tform)
        except Exception as e: 
            sample_dir = self.samples[0]
            target_im, cond_im, mask_target, mask_cond, metadata = get_sample(self.root_dir,sample_dir,self.tform)
        
        
        
        insert_loc = metadata["removed_object_camera_coords"]
        
        prompt = self.object_annot[metadata["removed_object_uid"]]["description"]
        prompt = "insert the {}".format(prompt)
        data["mask_cond"] = mask_cond
        data["mask_target"] = mask_target 
        data["uid"] = sample_dir
        data["image_target"] = target_im
        data["cond"] = {
            "image": cond_im,
            "text": prompt
        }
        data["T"] = self.get_T(insert_loc[0],insert_loc[1])
        data["log_key"] = self.log_key
        
        return data

    
class ObjaverseDataMultitask(Dataset):

    def __init__(
        self,
        root_dir='.DATASET',
        image_transforms=[],
        task="multitask",
        split="train",
        seen_or_unseen="seen",
        num_samples=100000,
    
    ):
        assert not (split == "train" and seen_or_unseen == "unseen")
        assert task == "multitask"

        self.tasks = ["rotate","remove","insert","translate"]

        self.task_classes = {
            "rotate": ObjaverseDataRotation,
            "translate": ObjaverseDataTranslation,
            "remove": ObjaverseDataRemove,
            "insert": ObjaverseDataInsert,

        }

        self.task_datasets = {
            t : self.task_classes[t](
                root_dir=root_dir,
                image_transforms=image_transforms,
                task=t,
                split=split,
                seen_or_unseen=seen_or_unseen,
                num_samples=num_samples
            ) for t in self.tasks
        }

        self.indices = []
        for t in self.tasks:
            task_len = len(self.task_datasets[t])
            self.indices += [(t,i) for i in range(task_len)]
            print(f'============= length of {t} task dataset {task_len} =============')

        self.len = len(self.indices)
        assert self.len == sum(len(self.task_datasets[t]) for t in self.task_datasets)
        print('============= length of multi-task dataset %d =============' % self.len)
    
    def __len__(self):

        return self.len

    def __getitem__(self,index):

        task, idx = self.indices[index]
        return self.task_datasets[task][idx]

# shared dataset functions

def preprocess_image(im,tform):
    im = im.convert("RGB")
    return tform(im)

def preprocess_mask(fp):
    mask = Image.open(fp).convert("RGBA") # by default they are saved w trivial alpha channel except removal
    mask = np.array(mask)
    mask_color_map = [
        (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),
        (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),
        (0.30196078431372547, 0.6862745098039216, 0.2901960784313726, 1.0),
        (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),
        (1.0, 0.4980392156862745, 0.0, 1.0),
        (1.0, 1.0, 0.2, 1.0),
        (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0),
        (0.9686274509803922, 0.5058823529411764, 0.7490196078431373, 1.0),
        (0.6, 0.6, 0.6, 1.0)
    ]

    mask_reshaped = mask.reshape(-1, 4)
    distances = np.linalg.norm(mask_reshaped[:, None, :] - mask_color_map, axis=-1)    
    min_indices = np.argmin(distances, axis=1)
    min_indices = min_indices.reshape(mask.shape[:2])

    mask = (min_indices == 1).astype(np.uint8)
    mask = torch.tensor(mask)

    # torchvision.utils.save_image(mask / 1.0,os.path.join("sample_masks",uid[:-4] + ".png"))

    return mask

     
def load_image(fp):
    img = plt.imread(fp)
    img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
    return img

def get_sample(root_dir,fp,tform):
    cond_im = preprocess_image(load_image(os.path.join(root_dir,fp,"1.png")),tform)
    target_im = preprocess_image(load_image(os.path.join(root_dir,fp,"2.png")),tform)
    mask_cond = preprocess_mask(os.path.join(root_dir,fp,"1_mask.png"))
    mask_target = preprocess_mask(os.path.join(root_dir,fp,"2_mask.png"))
    with open(os.path.join(root_dir,fp,"metadata.json"),"r") as f:
        metadata = json.load(f)

    return cond_im, target_im, mask_cond, mask_target, metadata

