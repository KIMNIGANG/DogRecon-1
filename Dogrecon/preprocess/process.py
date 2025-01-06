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



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--model', default='u2net', type=str, help="rembg model, see https://github.com/danielgatis/rembg#models")
    parser.add_argument('--size', default=512, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=str2bool, default=True, help="recenter, potentially not helpful for multiview zero123")    
    opt = parser.parse_args()

    session = rembg.new_session(model_name=opt.model)

    if os.path.isdir(opt.path):
        print(f'[INFO] processing directory {opt.path}...')
        files = glob.glob(f'{opt.path}/*')
        out_dir = opt.path
    else: # isfile
        files = [opt.path]
        out_dir = os.path.dirname(opt.path)
    
    for file in files:

        out_base = os.path.basename(file).split('.')[0]
        out_rgba = os.path.join(out_dir, out_base + '_rgba.png')
        out_resize = os.path.join(out_dir, out_base + '_resize.png')

        # load image
        print(f'[INFO] loading image {file}...')
        #image = cv2.imread(file, cv2.IMREAD_UNCHANGED) 원본
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        #image2 = cv2.resize(cv2.imread(file),(512,512))
        # carve background
        print(f'[INFO] background removal...')
        carved_image = rembg.remove(image, session=session) # [H, W, 4]
        mask = carved_image[..., -1] > 0

        # recenter
        if opt.recenter:
            print(f'[INFO] recenter...')

            final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
            #tttt = np.zeros((opt.size, opt.size, 3), dtype=np.uint8)
            tttt = np.ones((opt.size, opt.size, 3), dtype=np.uint8)*255
            coords = np.nonzero(mask)
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            h = x_max - x_min
            w = y_max - y_min
            desired_size = int(opt.size * (1 - opt.border_ratio))
            scale = desired_size / max(h, w)
            h2 = int(h * scale)
            w2 = int(w * scale)
            x2_min = (opt.size - h2) // 2
            x2_max = x2_min + h2
            y2_min = (opt.size - w2) // 2
            y2_max = y2_min + w2
            final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
            #print(Counter(list(carved_image[:,:,3].reshape(-1))))
            #tttt[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

            carved= carved_image[x_min:x_max, y_min:y_max][:,:,:3].copy()
            #carved[carved_image>0] = carved_image[x_min:x_max, y_min:y_max][:,:,:]
            #print(carved[carved_image[x_min:x_max, y_min:y_max][:,:,3]!=255], carved[0,0])
            #print(carved[carved_image[x_min:x_max, y_min:y_max][:,:,3]!=255], carved[carved_image[x_min:x_max, y_min:y_max][:,:,3]!=255].shape)
            carved[carved_image[x_min:x_max, y_min:y_max][:,:,3]<150] = [255,255,255]
            tttt[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved, (w2, h2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_resize,tttt)
        else:
            final_rgba = carved_image
        
        # write image
        
        
        cv2.imwrite(out_rgba, final_rgba)