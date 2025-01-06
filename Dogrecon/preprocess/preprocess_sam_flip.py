import os
import glob
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import rembg

from collections import Counter

from preprocess_utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave, resize_image
import os
from PIL import Image
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--image_path", required=True)
    parser.add_argument("--image_name", required=True)
    parser.add_argument("--ckpt_path", default="../GroundingDINO/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    # load SAM checkpoint
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    sam_predictor = sam_init(args.ckpt_path, gpu)
    print("load sam ckpt done.")

    input_raw = Image.open(f'./oneshot_image/{args.image_name}.png')
    input_raw = input_raw.convert('RGB')
    # input_raw.thumbnail([512, 512], Image.Resampling.LANCZOS)
    input_raw = resize_image(input_raw, 512)
    image_sam = sam_out_nosave(
        sam_predictor, input_raw, pred_bbox(input_raw)
    )

    image_preprocess(image_sam, args.image_name, lower_contrast=False, rescale=True)