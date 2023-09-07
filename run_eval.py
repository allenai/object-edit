import argparse
import os
import torch
import torchvision
from metrics import psnr_mask, ssim_mask, lpip_mask, fid
from ldm.util import instantiate_from_config
from ldm.data.simple import ObjaverseDataRotation, ObjaverseDataRemove, ObjaverseDataInsert, ObjaverseDataTranslation
from einops import rearrange
from tqdm import tqdm
import numpy as np
import json
import ipdb

task_classes = {
    "rotate": ObjaverseDataRotation,
    "translate": ObjaverseDataTranslation,
    "remove": ObjaverseDataRemove,
    "insert": ObjaverseDataInsert
}

def run(args):

    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor(),
    ])
    dataset = task_classes[args.task](
        root_dir=args.data_dir, 
        image_transforms=image_transforms,
        task=args.task,
        split=args.split,
        seen_or_unseen=args.seen,
    )
    sample_metrics = {
        "psnr_mask": psnr_mask,
        "ssim_mask": ssim_mask,
        "lpip_mask": lpip_mask,
    }
    sample_statistics = []
    best_samples = []
    fid_samples, fid_targets = [], []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        uid = sample["uid"]
        generated_samples = [torchvision.io.read_image(os.path.join(args.generation_dir,uid,f"{i}.png")) for i in range(4)]
        generated_samples = [s.unsqueeze(0) / 255. for s in generated_samples]
        mask_target = sample["mask_target"]
        image_target = sample["image_target"].unsqueeze(0)
        metrics_per_sample = [
            {k:f(s,image_target,mask_target,i,uid) for k,f in sample_metrics.items()} 
            for s in generated_samples
        ]
        sample_statistics.append(metrics_per_sample)
        best_sample_index = max(range(4),key=lambda x: metrics_per_sample[x]["psnr_mask"])
        best_samples.append(metrics_per_sample[best_sample_index])
        fid_samples.append((generated_samples[best_sample_index]*255).to(torch.uint8))
        fid_targets.append((image_target*255).to(torch.uint8))
    
    sample_averages = {}
    for k in sample_metrics.keys():
        sample_averages["mean_" + k] = np.mean([x[k] for x in best_samples])
    fid_samples, fid_targets = torch.cat(fid_samples,dim=0), torch.cat(fid_targets,dim=0)
    sample_averages["fid"] = fid(fid_samples,fid_targets)
    os.makedirs(args.save_dir,exist_ok=True)
    with open(os.path.join(args.save_dir,f"{args.task}_{args.split}_{args.seen}_summary.json"),"w") as f:
        json.dump(sample_averages,f,indent=2)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["rotate","remove","insert","translate"]
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train","val","test"]
    )
    parser.add_argument(
        "--seen",
        type=str,
        default="seen",
        choices=["seen","unseen"]
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="statistics"
    )
    args = parser.parse_args()
    assert not (args.split == "train" and args.seen == "unseen")
    run(args)
