#!/bin/bash
echo ========================================
echo 1/5: Stable Zero123
echo ========================================
conda activate prep
mkdir ./data/dog_data_official/dog/
mkdir ./data/dog_data_official/dog/images
mkdir ./data/dog_data_official/dog/pred
mkdir ./data/dog_data_official/dog/bak
python preprocess/preprocess_sam_flip.py --image_name dog
cp oneshot_image/dog_resize.png ./data/dog_data_official/dog/images/0000.png
cp oneshot_image/dog_resize2.png ./data/dog_data_official/dog/images/0072.png
cd ../zero123/zero123/
python gradio_new.py --image_name dog
python gradio_objaverse.py --image_name dog
echo ========================================
echo 2/5: GroundingDINO + SAM
echo ========================================
cd ../../GART/
python preprocess/do_mask.py --image_path dog --folder_int 0
cp ./data/dog_data_official/dog/images/* ./data/dog_data_official/dog/bak/
echo ========================================
echo 3/5: BITE
echo ========================================
cd ../bite_release/
conda deactivate
conda activate bite
python scripts/full_inference_including_ttopt.py --workers 12 --config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml --model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --suffix ttopt_vtest1 --image_path dog
python gptest_head_.py --image_name dog
python gptest_head_flip.py --image_name dog
conda deactivate
echo ========================================
echo 4/5: Geometric Prior
echo ========================================
conda activate prep
cd ../zero123/zero123/
python gradio_new.py --image_name dog
python gradio_objaverse_flip.py --image_name dog
cd ../../GART/
python preprocess/do_mask.py --image_path dog --folder_int 0
echo ========================================
echo 5/5: Gaussian Splatting
echo ========================================
cd ../GART/
conda deactivate
conda activate gart
python solver.py --profile ./profiles/dog/dog.yaml --dataset dog_demo --seq dog --logbase dog --no_eval --semantic clip --sampling mask
conda deactivate