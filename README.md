# DogRecon

## Overview

DogRecon is a pipeline for processing dog images through various stages including Zero123, GroundingDINO + SAM, BITE, Geometric Prior, and GART.

## Prerequisites

- Python 3.x
- Conda

## Setup

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd DogRecon
    ```

2. Create and activate the required conda environments as specified in the script.

## Usage

To generate a bash script for processing a dog image, run the `gen_run_ijcv.py` script with the `--image_name` argument:

```bash
python preprocess/gen_run_ijcv.py --image_name <path_to_image>
```

This will create a bash script named `run_<image_name>.sh` in the current directory.

## Example

```bash
python preprocess/gen_run_ijcv.py --image_name ./images/dog1.png
```

This command will generate a script `run_dog1.sh` that you can execute to process the image through the pipeline.

## Running the Generated Script

After generating the script, run it using:

```bash
bash run_<image_name>.sh
```

Replace `<image_name>` with the name of your image (without the file extension).

## Steps in the Pipeline

1. **Stable Zero123**: Initial processing and setup.
2. **GroundingDINO + SAM**: Mask generation and backup.
3. **BITE**: Full inference and optimization.
4. **Canine Geometric Prior**: Further processing using optimized masks.
5. **Gaussian Splatting(GART)**: Final processing to generate the output.



