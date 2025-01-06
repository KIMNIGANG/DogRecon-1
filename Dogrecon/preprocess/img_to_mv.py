import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import argparse
# Load the pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', required=True, type=str)
    opt = parser.parse_args()
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    # Feel free to tune the scheduler
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline.to('cuda:0')
    # Run the pipeline
    #cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)
    #image_name ='0719_rgba'
    cond = Image.open(f'/home/user/gs/huggstudy/GART/oneshot_image/{opt.image_name}_rgba.png')

    result = pipeline(cond, num_inference_steps=75).images[0]
    result.show()
    result.save(f"./oneshot_image/output_{opt.image_name}.png")

if __name__ =="__main__":
    main()

