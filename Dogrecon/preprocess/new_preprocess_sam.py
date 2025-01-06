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

import os, glob, argparse

import cv2
import numpy as np
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
#from segment_anything import sam_model_registry, SamPredictor



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
    TEXT_PROMPT = ["dog"]
    BOX_TRESHOLD = 0.30
    TEXT_TRESHOLD = 0.25

    print("load sam ckpt done.")

    input_raw = Image.open(f'./oneshot_image/{args.image_name}.png')
    input_raw = input_raw.convert('RGB')
    # input_raw.thumbnail([512, 512], Image.Resampling.LANCZOS)
    input_raw = resize_image(input_raw, 512)
    grounding_dino_model = Model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth") 
    numpy_image = np.array(input_raw)
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    #resized_image = cv2.resize(image_cv, (512, 512))

    detections = grounding_dino_model.predict_with_classes(
            image=cv2_image,
            classes=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD)
    
    image_sam = sam_out_nosave(
        sam_predictor, input_raw,detections.xyxy[0][0],detections.xyxy[0][1],detections.xyxy[0][2],detections.xyxy[0][3],
    )

    image_preprocess(image_sam, args.image_name, lower_contrast=False, rescale=True)