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
    commands.append(f'echo 1/{steps}: Stable Zero123')
    commands.append(f'echo ========================================')
    commands.append('conda activate prep')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/images')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}/pred')

    commands.append(f'mkdir ./data/dog_data_official/{image_name}_gart/')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}_gart/images')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}_gart/pred')

    commands.append(f'mkdir ./data/dog_data_official/{image_name}_naivegeo/')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}_naivegeo/images')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}_naivegeo/pred')


    commands.append(f'mkdir ./data/dog_data_official/{image_name}_caninegeo/')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}_caninegeo/images')
    commands.append(f'mkdir ./data/dog_data_official/{image_name}_caninegeo/pred')

    commands.append(f'python preprocess/preprocess_sam.py --image_name {image_name}')
    commands.append(f'cp oneshot_image/{image_name}_resize.png ./data/dog_data_official/{image_name}/images/0000.png')
    commands.append(f'cd ../zero123/zero123/')
    commands.append(f'python stablezero123.py --image_name {image_name}')
    
    commands.append(f'echo ========================================')
    commands.append(f'echo 2/{steps}: GroundingDINO + SAM')
    commands.append(f'echo ========================================')
    commands.append(f'cd ../../GART/')
    commands.append(f'python preprocess/do_mask.py --image_path {image_name} --folder_int 0')

    commands.append(f'cp ./data/dog_data_official/{image_name}/images/* ./data/dog_data_official/{image_name}_gart/images')
    commands.append(f'cp ./data/dog_data_official/{image_name}/images/* ./data/dog_data_official/{image_name}_naivegeo/images')
    commands.append(f'cp ./data/dog_data_official/{image_name}/images/* ./data/dog_data_official/{image_name}_caninegeo/images')

    commands.append(f'echo ========================================')
    commands.append(f'echo 3/{steps}: BITE')
    commands.append(f'echo ========================================')
    commands.append('cd ../bite_release/')
    commands.append('conda deactivate')
    commands.append('conda activate bite')
    commands.append(f'python scripts/full_inference_including_ttopt.py --workers 12 --config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml --model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --suffix ttopt_vtest1 --image_path {image_name}')
    
    commands.append(f'cp ../GART/data/dog_data_official/{image_name}/pred/* ../GART/data/dog_data_official/{image_name}_gart/pred')
    commands.append(f'cp ../GART/data/dog_data_official/{image_name}/pred/* ../GART/data/dog_data_official/{image_name}_naivegeo/pred')
    commands.append(f'cp ../GART/data/dog_data_official/{image_name}/pred/* ../GART/data/dog_data_official/{image_name}_caninegeo/pred')


    commands.append(f'python geometricprior.py --image_name {image_name}_naivegeo')


    commands.append(f'echo ========================================')
    commands.append(f'echo 4/{steps}: Gaussian Splatting')
    commands.append(f'echo ========================================')
    commands.append('cd ../GART/')
    commands.append('conda deactivate')
    commands.append('conda activate gart')
    commands.append(f'python solver.py --profile ./profiles/dog/dog.yaml --dataset dog_demo --seq {image_name} --logbase dog --no_eval --semantic clip')
    commands.append(f'python solver.py --profile ./profiles/dog/dog.yaml --dataset dog_demo --seq {image_name}_gart --logbase dog --no_eval --semantic clip')
    commands.append(f'python solver.py --profile ./profiles/dog/dog.yaml --dataset dog_demo --seq {image_name}_naivegeo --logbase dog --no_eval --semantic clip')
    commands.append('conda deactivate')



    print(*commands, sep='\n')
    with open(f"run_{image_name}.sh", "w") as outfile:
        outfile.write("\n".join(commands))

if __name__ == "__main__":
    main()