"""
Do_mask.py



"""
import os, glob, argparse

import cv2
import numpy as np
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from segment_anything import sam_model_registry, SamPredictor

#python do_makemasek.py --image_path oneshot_york --folder_int 0

folder_list = ['/images','/pred']

# Hyperparameters for SAM
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH ='/home/user/gs/huggstudy/GroundingDINO/sam_vit_h_4b8939.pth'

# Call SAM
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=None)
   #parser.add_argument("--keypoints_path", type=str, default=None)
    parser.add_argument("--folder_int", type=int)
    
    args = parser.parse_args()
    image_path= args.image_path
    folder_int = args.folder_int
    #IMAGE_PATH = "/root/share2/animal/animal_preprocess/total_v2_resize/raw_720p/00000.png"
    TEXT_PROMPT = ["dog"]
    BOX_TRESHOLD = 0.30
    TEXT_TRESHOLD = 0.25

    # Call DINO
    grounding_dino_model = Model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth") 

    save_path = f"./data/dog_data_official/{image_path}"+folder_list[folder_int]
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #print(save_path)
    img_lists = sorted(glob.glob(f"{save_path}/*.png"))
    
    # Can be deleted
    SAM_CHECKPOINT_PATH ='/home/user/gs/huggstudy/GroundingDINO/sam_vit_h_4b8939.pth'

    # Call SAM
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to("cuda") #.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    print(img_lists)

    for fn in tqdm(img_lists):
        image = cv2.imread(fn)

        # Do DINO
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD)
       # print(detections.xyxy,detections.xyxy.shape)
        #[[ 47.24106   79.020004 451.4621   447.8526  ]] (1, 4)
        # Do Segmentations
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy)

        mask_image = np.uint8((detections.mask[0]))
        print(fn.split('/')[-1][:-4])
        np.save(f"{fn[:-4]}.npy",mask_image)
        #print(mask_image, mask_image.shape)