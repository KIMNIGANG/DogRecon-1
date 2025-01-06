import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', required=True, type=str, help='the path to the source video')

    opt = parser.parse_args()
    image_name = opt.image_name #.split('/')[-1].split('.')[0]

    commands = []
    commands.append('#!/bin/bash')
    #steps = 5
    #commands.append(f'echo ========================================')
    #commands.append(f'echo 1/{steps}: Stable Zero123')
    #commands.append(f'echo ========================================')

    for i in range(50):  # 66개의 명령어 생성
        #command = f"python preprocess/gen_run_flip.py --image_name ./oneshot_image/d{i}_flip.png"
        command=f'mv ./{image_name}/{i:04d} ./{image_name}/{i+174:04d}'
        commands.append(command)










    print(*commands, sep='\n')
    with open(f"run_shsh_{image_name}.sh", "w") as outfile:
        outfile.write("\n".join(commands))

if __name__ == "__main__":
    main()