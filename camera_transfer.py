import os
import subprocess
import numpy as np
import json
from PIL import Image
import pdb

def quaternion_to_rotation_matrix(quaternion):
    q_w, q_x, q_y, q_z = quaternion
    R = np.array([[1 - 2*q_y**2 - 2*q_z**2, 2*q_x*q_y - 2*q_z*q_w, 2*q_x*q_z + 2*q_y*q_w],
        [2*q_x*q_y + 2*q_z*q_w, 1 - 2*q_x**2 - 2*q_z**2, 2*q_y*q_z - 2*q_x*q_w],
        [2*q_x*q_z - 2*q_y*q_w, 2*q_y*q_z + 2*q_x*q_w, 1 - 2*q_x**2 - 2*q_y**2]])
    return R

def moving_camera(cam_path, output_path):
    img_add = os.path.join(cam_path, 'camera/images.txt')
    cam_add = os.path.join(cam_path, 'camera/cameras.txt')
    DICTION = {}
    with open(cam_add, "r") as file:
        lines = file.readlines()
        intrinsic = lines[3].split(" ")[4:]
        focal1 = intrinsic[0]
        focal2 = intrinsic[1]
        cx = intrinsic[2]
        cy = intrinsic[3]
        #distortion = intrinsic[4]
        INT = np.eye(3)
        INT[0, 0] = focal1
        INT[1, 1] = focal2
        INT[0, 2] = cx
        INT[1, 2] = cy
        DICTION[f"int"] = INT.tolist()
        #DICTION[f"distortion"] = distortion
    with open(img_add, "r") as file:
        lines = file.readlines()
        lines = lines[4:]
        lines = lines[::2]
        for line in lines:
            # Do something with each even line (e.g., print it)
            trans = line.strip().split(" ")[5:8]  # Strip whitespace and newline characters
            trans = np.array([float(T) for T in trans])
            quant = line.strip().split(" ")[1:5]
            quant = np.array([float(T) for T in quant])
            rotation = quaternion_to_rotation_matrix(-quant)
            c2w = np.eye(4)
            c2w[:3, :3] = rotation
            c2w[:3, 3] = trans
            name = int(line.strip().split(" ")[-1].split('.')[0][6:])
            DICTION[f"ext_{name}"] = c2w.tolist()
    with open(output_path, "w") as json_file:
        json.dump(DICTION, json_file)

def moving_colmap(_input_, _output_, category_list):
    out_add1 = _input_
    out_add2 = _output_
    for category in category_list:
        category_dir = os.path.join(out_add1, category)
        sub_category_list = [d for d in os.listdir(category_dir)]
        for sub_category in sub_category_list:
            directory = os.path.join(category_dir, sub_category)
            cam_out = os.path.join(out_add2, category, f"{sub_category}_camera_pose.json")
            moving_camera(directory, cam_out)

def convert(pre_json, name_list, save_dir):
    with open(pre_json, 'r') as file:
        data_dict = json.load(file)
    save_dict = {}
    intrinsic = np.array(data_dict["int"])
    save_dict["camera_angle_x"] = 2 * np.arctan(intrinsic[0,-1]/intrinsic[0,0])
    save_dict["camera_angle_y"] = 2 * np.arctan(intrinsic[1,-1]/intrinsic[1,1])
    save_dict["frames"] = []
    for name in name_list:
        index_img = int(name.split('.')[0].split('_')[1])
        extrinsic_name = f"ext_{index_img}"
        extrinsic_matrix = np.array(data_dict[extrinsic_name])
        c2w = np.linalg.inv(extrinsic_matrix)
        c2w[0:3,2] *= -1 
        c2w[0:3,1] *= -1
        c2w = c2w[[1,0,2,3],:]
        c2w[2,:] *= -1 
        current_dict = {}
        current_dict["file_path"] = name
        current_dict["transform_matrix"] = c2w.tolist()
        save_dict["frames"].append(current_dict)
    file_path = save_dir
    print(file_path)
    with open(file_path, 'w') as json_file:
        json.dump(save_dict, json_file)

def convert_gaussian(pref_input):
    category_list = [d for d in os.listdir(pref_input)]
    for category in category_list:
        category_dir = os.path.join(pref_input, category)
        sub_category_list = [d for d in os.listdir(category_dir)]
        for sub_category in sub_category_list:
            if not sub_category.endswith('json'):
                pre_trans_json = os.path.join(category_dir, sub_category+"_camera_pose.json")
                save_dir = os.path.join(category_dir, sub_category+"_gaussian_pose.json")
                image_add = os.path.join(category_dir, sub_category, 'dense/images')
                name_list = [f for f in os.listdir(image_add) if f.endswith('.png')]
                convert(pre_trans_json, name_list, save_dir)

if __name__ == "__main__":
    step = 1
    if step == 1:
        # Moving precomputed colmap pose to .json structure
        _input_ = "example_refine"
        _output_ = "example_refine" 
        category_list = ["rubberduck"]
        moving_colmap(_input_, _output_, category_list)
    if step == 2:
        # transform .json structure to 3D gaussian camera structure
        pref_input = "example_refine"
        convert_gaussian(pref_input)