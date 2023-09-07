import torchmetrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import torchvision

lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

def fix_bounds(xmin,xmax,ymin,ymax,size,min_size=32):

    if xmax - xmin < min_size: # if size of masked region is too small, we make size 64
        center = (xmax + xmin) // 2
        xmin = max(center - min_size,0)
        xmax = min(center + min_size,size-1)
        if xmin == 0:
            xmax += min_size - (xmax - xmin)
        elif xmax == size-1:
            xmin -= min_size - (xmax - xmin)

    if ymax - ymin < min_size:
        center = (ymax + ymin) // 2
        ymin = max(center - min_size,0)
        ymax = min(center + min_size,size-1)
        if xmin == 0:
            ymax += min_size - (ymax - ymin)
        elif xmax == size-1:
            ymin -= min_size - (ymax - ymin)

    return xmin,xmax,ymin,ymax


def psnr(x,y,*args):

    return peak_signal_noise_ratio(x,y).item()

def psnr_mask(x,y,mask,*args):

    rows, cols = torch.where(mask)
    if len(rows) > 0:
        xmin, ymin = rows.min().item(), cols.min().item()
        xmax, ymax = rows.max().item(), cols.max().item()

        xmin, xmax, ymin, ymax = fix_bounds(xmin,xmax,ymin,ymax,256)

        x_region = x[:,:, xmin:xmax, ymin:ymax]
        y_region = y[:,:, xmin:xmax, ymin:ymax]
    else: # object fully occluded
        x_region, y_region = x ,y

    return psnr(x_region,y_region)

    

def ssim(x,y,*args):
    
    return structural_similarity_index_measure(x,y).item()

def ssim_mask(x,y,mask,*args):

    rows, cols = torch.where(mask)
    if len(rows) > 0:
        xmin, ymin = rows.min().item(), cols.min().item()
        xmax, ymax = rows.max().item(), cols.max().item()

        xmin, xmax, ymin, ymax = fix_bounds(xmin,xmax,ymin,ymax,256)

        x_region = x[:,:, xmin:xmax, ymin:ymax]
        y_region = y[:,:, xmin:xmax, ymin:ymax]
    else: # object fully occluded
        x_region, y_region = x,y

    return ssim(x_region,y_region)
    
    
@torch.no_grad()
def lpip(x,y,*args):

    x = 2*x - 1
    y = 2*y - 1
    return -lpips(x,y).item()

@torch.no_grad()
def lpip_mask(x,y,mask,*args):

   
    rows, cols = torch.where(mask)
    if len(rows) > 0:
        xmin, ymin = rows.min().item(), cols.min().item()
        xmax, ymax = rows.max().item(), cols.max().item()

        xmin, xmax, ymin, ymax = fix_bounds(xmin,xmax,ymin,ymax,256)

        x_region = x[:,:, xmin:xmax, ymin:ymax]
        y_region = y[:,:, xmin:xmax, ymin:ymax]
    else: # object fully occluded
        x_region, y_region = x, y
   
    return lpip(x_region,y_region)

def fid(xs,ys,*args):

    fid_metric = FrechetInceptionDistance(feature=64)
    fid_metric.update(xs,real=False)
    fid_metric.update(ys,real=True)

    return fid_metric.compute().item()


