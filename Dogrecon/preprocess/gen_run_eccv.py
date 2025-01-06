import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', required=True, type=str, help='the path to the source video')

    opt = parser.parse_args()
    image_name = opt.image_name.split('/')[-1].split('.')[0]

    commands = []
    commands.append('#!/bin/bash')
    steps = 5
    commands.append(f'echo ========================================')
    commands.append(f'echo 1/{steps}: BITE-1')
    commands.append(f'echo ========================================')
    commands.append('conda activate prep')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/images')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/pred')

    commands.append(f'mkdir ./data/dog_data_official/{image_name}/images_temp')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/pred_temp')


    commands.append(f'python preprocess/preprocess_sam_flip.py --image_name {image_name}')
    commands.append(f'cp oneshot_image/{image_name}_resize.png ./data/dog_data_official/{image_name}/images/0000.png')
    commands.append(f'cp oneshot_image/{image_name}_resize2.png ./data/dog_data_official/{image_name}/images/0072.png')
    #commands.append(f'cd ../zero123/zero123/')
    #commands.append(f'python stablezero123.py --image_name {image_name}')
    #commands.append(f'python stablezero123_flip.py --image_name {image_name}')
    
    #commands.append(f'echo ========================================')
    #commands.append(f'echo 2/{steps}: GroundingDINO + SAM')
    #commands.append(f'echo ========================================')
    #commands.append(f'cd ../../GART/')
    #commands.append(f'python preprocess/do_mask.py --image_path {image_name} --folder_int 0')
    #commands.append(f'cp ./data/dog_data_official/{image_name}/images/* ./data/dog_data_official/{image_name}/bak/')


    commands.append(f'echo ========================================')
    commands.append(f'echo 3/{steps}: BITE')
    commands.append(f'echo ========================================')
    commands.append('cd ../bite_release/')
    commands.append('conda deactivate')
    commands.append('conda activate bite')
    commands.append(f'python scripts/full_inference_including_ttopt.py --workers 12 --config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml --model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --suffix ttopt_vtest1 --image_path {image_name}')
    
    commands.append(f'python geometricprior_eccv.py --image_name {image_name}') #여기서 나온값으로 stalbe zero123의 output을 옮겨줘야함. 값 저장해야함. 
    commands.append(f'python geometricprior_eccv_flip.py --image_name {image_name}')
    commands.append('conda deactivate')
    commands.append(f'echo ========================================')
    commands.append(f'echo 4/{steps}: Geometric Prior')
    commands.append(f'echo ========================================')

    
    commands.append('conda activate prep')
    commands.append(f'cd ../zero123/zero123/')
    commands.append(f'python stablezero123.py --image_name {image_name}')
    commands.append(f'python stablezero123_flip.py --image_name {image_name}')

    # images_temp로 옮기는 과정 추가.
    # python movetotemp
    commands.append('cd ../../bite_release/')
    commands.append('conda deactivate')
    commands.append('conda activate bite')
    commands.append(f'python move2temp.py --image_name {image_name}')
    commands.append(f'python move2temp_flip.py --image_name {image_name}')
    commands.append(f'python scripts/full_eccv.py --workers 12 --config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml --model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --suffix ttopt_vtest1 --image_path {image_name}')
    
    # pred로 옮기는과정 추가.
    commands.append(f'cp ../GART/data/dog_data_official/{image_name}/images_temp/* ../GART/data/dog_data_official/{image_name}/images/')
    commands.append(f'cp ../GART/data/dog_data_official/{image_name}/pred_temp/* ../GART/data/dog_data_official/{image_name}/pred/')

    commands.append(f'cd ../../GART/')
    commands.append(f'python preprocess/do_mask.py --image_path {image_name} --folder_int 0')

    commands.append(f'echo ========================================')
    commands.append(f'echo 5/{steps}: Gaussian Splatting')
    commands.append(f'echo ========================================')
    commands.append('cd ../GART/')
    commands.append('conda deactivate')
    commands.append('conda activate gart')
    commands.append(f'python solver.py --profile ./profiles/dog/dog.yaml --dataset dog_demo --seq {image_name} --logbase dog --no_eval --semantic clip --sampling mask')
    commands.append('conda deactivate')



    print(*commands, sep='\n')
    with open(f"run_{image_name}.sh", "w") as outfile:
        outfile.write("\n".join(commands))

if __name__ == "__main__":
    main()