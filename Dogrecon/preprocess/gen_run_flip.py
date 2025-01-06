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
    commands.append(f'echo 1/{steps}: Stable Zero123')
    commands.append(f'echo ========================================')
    commands.append('conda activate prep')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/images')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/pred')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/bak')


    commands.append(f'python preprocess/preprocess_sam_flip.py --image_name {image_name}')
    commands.append(f'cp oneshot_image/{image_name}_resize.png ./data/dog_data_official/{image_name}/images/0000.png')
    commands.append(f'cp oneshot_image/{image_name}_resize2.png ./data/dog_data_official/{image_name}/images/0072.png')
    commands.append(f'cd ../zero123/zero123/')
    commands.append(f'python stablezero123.py --image_name {image_name}')
    commands.append(f'python stablezero123_flip.py --image_name {image_name}')
    
    commands.append(f'echo ========================================')
    commands.append(f'echo 2/{steps}: GroundingDINO + SAM')
    commands.append(f'echo ========================================')
    commands.append(f'cd ../../GART/')
    commands.append(f'python preprocess/do_mask.py --image_path {image_name} --folder_int 0')
    commands.append(f'cp ./data/dog_data_official/{image_name}/images/* ./data/dog_data_official/{image_name}/bak/')


    commands.append(f'echo ========================================')
    commands.append(f'echo 3/{steps}: BITE')
    commands.append(f'echo ========================================')
    commands.append('cd ../bite_release/')
    commands.append('conda deactivate')
    commands.append('conda activate bite')
    #pred ... 이건 그냥 zero123의 bite result
    commands.append(f'python scripts/full_inference_including_ttopt.py --workers 12 --config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml --model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --suffix ttopt_vtest1 --image_path {image_name}')
    
    #pred ... 이건 optimization한 SMAL의 bite result
    commands.append(f'python geometricprior_test_72_only0.py --image_name {image_name}')
    commands.append(f'python geometricprior_test_flip_only0.py --image_name {image_name}')
    commands.append('conda deactivate')
    commands.append(f'echo ========================================')
    commands.append(f'echo 4/{steps}: Geometric Prior')
    commands.append(f'echo ========================================')

    
    commands.append('conda activate prep')
    commands.append(f'cd ../zero123/zero123/')
    #optimization된 새로운 SMAL mask사용하여 mask-guided-zero123
    commands.append(f'python stablezero123.py --image_name {image_name}')
    commands.append(f'python stablezero123_flip.py --image_name {image_name}')
    commands.append(f'cd ../../GART/')
    commands.append(f'python preprocess/do_mask.py --image_path {image_name} --folder_int 0')

    commands.append(f'echo ========================================')
    commands.append(f'echo 5/{steps}: Gaussian Splatting')
    commands.append(f'echo ========================================')
    commands.append('cd ../GART/')
    commands.append('conda deactivate')
    commands.append('conda activate gart')
    #RSW를 거쳐서 최종 gaussian-dog 생성
    commands.append(f'python solver.py --profile ./profiles/dog/dog.yaml --dataset dog_demo --seq {image_name} --logbase dog --no_eval --semantic clip --sampling mask')
    commands.append('conda deactivate')



    print(*commands, sep='\n')
    with open(f"run_{image_name}.sh", "w") as outfile:
        outfile.write("\n".join(commands))

if __name__ == "__main__":
    main()