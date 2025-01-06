import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', required=True, type=str, help='the path to the source video')

    opt = parser.parse_args()
    image_name = opt.image_name.split('/')[-1].split('.')[0]

    commands = []
    commands.append('#!/bin/bash')
    steps = 4
    commands.append(f'echo ========================================')
    commands.append(f'echo 1/{steps}: Zero123++')
    commands.append(f'echo ========================================')
    commands.append('conda activate mask')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/images')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/pred')

    commands.append(f'python preprocess/process.py oneshot_image/{image_name}.png')
    commands.append(f'cp oneshot_image/{image_name}_resize.png ./data/dog_data_official/{image_name}/images/0000.png')
    commands.append(f'python preprocess/img_to_mv.py --image_name {image_name}')
    commands.append(f'python preprocess/split_and_resize.py --image_name {image_name}')
    
    commands.append(f'echo ========================================')
    commands.append(f'echo 2/{steps}: GroundingDINO + SAM')
    commands.append(f'echo ========================================')
    commands.append(f'python preprocess/do_mask.py --image_path {image_name} --folder_int 0')

    commands.append(f'echo ========================================')
    commands.append(f'echo 3/{steps}: BITE')
    commands.append(f'echo ========================================')
    commands.append('cd ../bite_release/')
    commands.append(f'python scripts/full_inference_including_ttopt.py --workers 12 --config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml --model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --suffix ttopt_vtest1 --image_path {image_name}')
    

    commands.append(f'echo ========================================')
    commands.append(f'echo 4/{steps}: Gaussian Splatting')
    commands.append(f'echo ========================================')
    commands.append('cd ../GART/')
    commands.append('conda deactivate')
    commands.append('conda activate gart')
    commands.append(f'python solver.py --profile ./profiles/dog/dog.yaml --dataset dog_demo --seq {image_name} --logbase dog --no_eval')
    commands.append('conda deactivate')



    print(*commands, sep='\n')
    with open(f"run_{image_name}.sh", "w") as outfile:
        outfile.write("\n".join(commands))

if __name__ == "__main__":
    main()