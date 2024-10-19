import cv2
import numpy as np
import os
from glob import glob
import argparse


def get_last_number_from_filename_image(filename):
    # Extract the last number from the filename
    numbers = (filename.split("/")[-1])[:-4].split("_")[1]
    return int(numbers)

def get_last_number_from_filename_mask(filename):
    # Extract the last number from the filename
    numbers = (filename.split("/")[-1])[:-4].split("_")[1]
    return int(numbers)

def mask_image(dir, dir_output):
    # Get list of files in the directory with modification times
    image_list = glob(os.path.join(dir, '*.png'))
    image_list = sorted(image_list, key=get_last_number_from_filename_image)
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    for number, image_dir in enumerate(image_list):
        # Load the original image and the mask
        original_image = cv2.imread(image_dir)
        name = image_dir.split('/')[-1]
        cv2.imwrite(os.path.join(dir_output, f'{name}.png'), original_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='exhaustive_matcher')
    parser.add_argument('--dir_output', type=str, default='exhaustive_matcher')
    args = parser.parse_args()
    mask_image(args.dir, args.dir_output)
    # change the name of mask.png to mask.png.png