from PIL import Image
import argparse

def split_and_super_resolution(image_path, output_path, original_size=(640, 960), new_size=(512, 512)):
    # Open the image
    img = Image.open(image_path)
    size = (320,320)

    # Get the dimensions of the image
    width, height = img.size

    # Calculate the number of rows and columns
    rows = height // size[1]
    cols = width // size[0]

    # Split the image into parts and perform super-resolution
    i = 0
    for row in range(rows):
        for col in range(cols):
            i +=1

            left = col * size[0]
            top = row * size[1]
            right = left + size[0]
            bottom = top + size[1]

            # Crop the original part
            original_part = img.crop((left, top, right, bottom))

            # Resize the part to 512x512 for super-resolution
            super_res_part = original_part.resize(new_size, Image.BICUBIC)  # You can use other resampling filters as well

            # Save the super-resolution part
            super_res_part.save(f"{output_path}/{i:04d}.png")

if __name__ == "__main__":
    # Specify your input and output paths
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', required=True, type=str, help='the path to the source video')

    opt = parser.parse_args()

    input_image_path = os.path.join('oneshot_image',f'output_{opt.image_name}.png')
    output_directory = os.path.join('data','dog_data_official',opt.image_name,'images')

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Split the image and perform super-resolution
    split_and_super_resolution(input_image_path, output_directory)