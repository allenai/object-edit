import argparse
import os
import torch
import torchvision
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt
from einops import rearrange
import math
import numpy as np
from PIL import Image

base_path = "/net/nfs/prior/oscarm/best_checkpoints_2.0"

def load_checkpoint(model,checkpoint):

    print(f"Attempting to load state from {checkpoint}")
    old_state = torch.load(checkpoint, map_location="cpu")

    if "state_dict" in old_state:
        print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
        old_state = old_state["state_dict"]

    # Check if we need to port weights from 4ch input to 8ch
    in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
    new_state = model.state_dict()
    in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
    in_shape = in_filters_current.shape
    if in_shape != in_filters_load.shape:
        input_keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model_ema.diffusion_modelinput_blocks00weight",
        ]
        
        for input_key in input_keys:
            if input_key not in old_state or input_key not in new_state:
                continue
            input_weight = new_state[input_key]
            if input_weight.size() != old_state[input_key].size():
                print(f"Manual init: {input_key}")
                input_weight.zero_()
                input_weight[:, :4, :, :].copy_(old_state[input_key])
                old_state[input_key] = torch.nn.parameter.Parameter(input_weight)

    m, u = model.load_state_dict(old_state, strict=False)

    if len(m) > 0:
        print("missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)

def preprocess_image(img):

    img = img.convert("RGB")
    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
    ])
    img = image_transforms(img)
    img = img*2. - 1.
    img = img.unsqueeze(0)
    return img

def sample_model(
    model,
    input_im,
    prompt,
    T,
    sampler,
    ddim_steps,
    n_samples,
    scale=1.0,
    ddim_eta=1.0
):

    print(prompt)
    c = model.get_learned_conditioning({"image":input_im,"text":prompt}).tile(n_samples, 1, 1)
    null_prompt = model.get_learned_conditioning([""])
    uc = null_prompt.repeat(1,c.shape[1],1).tile(n_samples,1,1)
    T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
    T = T.repeat(1,c.shape[1],1)
    c = torch.cat([c, T], dim=-1)
    c = model.cc_projection(c)
    cond = {}
    cond['c_crossattn'] = [c]
    cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                        .repeat(n_samples, 1, 1, 1)]
    uncond = {}
    uncond['c_crossattn'] = [uc]
    uncond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                        .repeat(n_samples, 1, 1, 1)]
    h, w = 256, 256
    shape = [4, h // 8, w // 8]
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        conditioning=cond,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uncond,
                                        eta=ddim_eta,
                                        x_T=None)

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp(x_samples_ddim, -1. ,1.)
    x_samples_ddim = ((x_samples_ddim + 1.0) / 2.0).cpu()

    return x_samples_ddim

def run(args):

    print("LOADING MODEL!")
    config = OmegaConf.load(f"configs/sd-objaverse-{args.task}.yaml")
    OmegaConf.update(config,"model.params.cond_stage_config.params.device",args.device)
    model = instantiate_from_config(config.model)
    model.cpu()
    load_checkpoint(model,args.checkpoint_path)
    model.to(args.device)
    model.eval()
    print("FINISHED LOADING!")

    image = Image.open(args.image_path)
    input_im = preprocess_image(image).to(args.device)
    x, y = map(float, args.position.split(','))
    if args.task == "rotate":
        prompt = f"rotate the {args.object_prompt}"
        azimuth = math.radians(args.rotation_angle)
        T = torch.tensor([np.pi / 2, math.sin(azimuth), math.cos(azimuth),0])
    elif args.task == "remove":
        prompt = f"remove the {args.object_prompt}"
        T = torch.tensor([0.,0.,0.,0.])
    elif args.task == "insert":
        prompt = f"insert the {args.object_prompt}"
        T = torch.tensor([0,x,y,0])
    elif args.task == "translate":
        prompt = f"move the {args.object_prompt}"
        T = torch.tensor([0,x,y,0])
    

    sampler = DDIMSampler(model)

    x_samples_ddim = sample_model(
        model,
        input_im,
        prompt,
        T,
        sampler,
        args.ddim_steps,
        args.num_samples,
        scale=args.cfg_scale
    )
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    input_im = ((input_im + 1.0) / 2.0).cpu()[0]
    input_im = 255.0 * rearrange(input_im.numpy(), 'c h w -> h w c')
    input_im = Image.fromarray(input_im.astype(np.uint8))

    os.makedirs(args.save_dir,exist_ok=True)

    input_im.save(os.path.join(args.save_dir,"input_im.png"))
    for i,img in enumerate(output_ims):
        img.save(os.path.join(args.save_dir,f"{i}.png"))

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["rotate","remove","insert","translate"]
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="generated_images"
    )
    parser.add_argument(
        "--object_prompt",
        type=str,
        required=True
    )
    parser.add_argument(
        "--rotation_angle",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--position",
        type=str,
        default="0.5,0.5",
        help="Coordinates in x,y form where 0 <= x,y <= 1"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0
    )
    args = parser.parse_args()
    run(args)

    
