# Implementation of P-Shuffle in the paper "Understanding and Improving Robustness of Vision Transformers through Patch-based Negative Augmentation"
# The link to this paper: https://arxiv.org/pdf/2110.07858.pdf
import os
from glob import glob
import torch
from torchvision import transforms
import copy
from PIL import Image
import copy
import numpy as np
import math
import cv2
import random

class P_Shuffle(object):
    def __init__(self, patch_size=16):  # The default values are determined in assumption that the size of input image is 224*224.
        self.patch_size = patch_size

    def __call__(self, img):    # img must be a tensor whose shape has the form of (C, H, W).
                                # img must be a tensor to which all the transforms which can affect the tensor's shape (e.g. Resize, RandomCrop, etc.) are applied.
        temp = torch.zeros_like(img)
        
        patches, n_patch_vertical, n_patch_horizontal, height_patch, width_patch = split_and_arrange_img(img, self.patch_size)
        random.shuffle(patches)
        
        for idx, patch in enumerate(patches):
            patch_idx_vertical = idx // n_patch_horizontal
            patch_idx_horizontal = idx % n_patch_horizontal
            temp[:, patch_idx_vertical*height_patch : (patch_idx_vertical+1)*height_patch, patch_idx_horizontal*width_patch : (patch_idx_horizontal+1)*width_patch] = patch

        resultant_img = temp
        return resultant_img

def split_and_arrange_img(img, patch_size):
    h, w = img.shape[-2:]
    
    if isinstance(patch_size, int) and h % patch_size==0 and w % patch_size==0:
        h_patch = w_patch = patch_size
        num_patch_vertical = h // patch_size
        num_patch_horizontal = w // patch_size
        num_patch = num_patch_vertical * num_patch_horizontal
    elif isinstance(patch_size, (tuple, list)) and len(patch_size)==2 and h % patch_size==0 and w % patch_size==0:
        h_patch = patch_size[0]
        w_patch = patch_size[1]
        num_patch_vertical = h // h_patch
        num_patch_horizontal = w // w_patch
        num_patch = num_patch_vertical * num_patch_horizontal
    else: raise Exception("Please check the size of image or patch_size")
    
    patches = []
    for i in range(num_patch_vertical):
        for j in range(num_patch_horizontal):
            single_patch = img[:, i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch]
            patches.append(single_patch)
    return patches, num_patch_vertical, num_patch_horizontal, h_patch, w_patch
    
    
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='testing p-shuffle')
    parser.add_argument('--resize', default=640, type=int)
    parser.add_argument('--patch_size', default=32, type=int)
    parser.add_argument('--img_path', default='./1538130_3.jpg')
    parser.add_argument('--save_dir', default='./runs/p-shuffle')
    parser.add_argument('--result_name', default='result.jpg')
    
    args = parser.parse_args()
    print(args)
    
    transform = transforms.Compose([transforms.PILToTensor(),
                                    transforms.Resize((args.resize, args.resize)),
                                    P_Shuffle(args.patch_size),
                                    transforms.ToPILImage()])
    
    img = Image.open(args.img_path)
    img_trans = transform(img)
    
    os.makedirs(args.save_dir, exist_ok=True)
    img_trans.save(os.path.join(args.save_dir, args.result_name))
    