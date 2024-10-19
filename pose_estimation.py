import pdb 
import os
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import subprocess
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

# Function to adjust brightness for COLMAP (range: -50 to 50 recommended)
def adjust_brightness(image, brightness=30):
    return cv2.convertScaleAbs(image, alpha=1, beta=brightness)
# Function to adjust contrast for COLMAP (range: 0.8 to 1.5 recommended)
def adjust_contrast(image, contrast=1.5):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=0)
# Function to apply gamma correction for COLMAP (range: 0.8 to 1.2 recommended)
def adjust_gamma(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def create_preprocess(input_add, out_add, category_list, sub_category_list):
    input_add =    os.path.join(input_add, "final_data_2024")
    input_add_msk =    os.path.join(input_add, "final_data_2024_mask")
    out_add_img =  os.path.join(out_add, "final_data_2024")
    out_add_msk =  os.path.join(out_add, "final_data_2024_mask")

    out_add_img_bri1 = out_add_img + "_bri1"
    out_add_img_con1 = out_add_img + "_con1"
    out_add_img_gam1 = out_add_img + "_gam1"
    out_add_img_bri2 = out_add_img + "_bri2"
    out_add_img_con2 = out_add_img + "_con2"
    out_add_img_gam2 = out_add_img + "_gam2"
    aug_list = [out_add_img, out_add_img_bri1, out_add_img_con1, out_add_img_gam1, \
        out_add_img_bri2, out_add_img_con2, out_add_img_gam2]
    for aug_add in aug_list:
        if not os.path.exists(aug_add):
            os.makedirs(aug_add)
    category_list = ["rubberduck"]
    for category in category_list:
        sub_category_list = ['before_missing']
        for sub_category in sub_category_list:
            for aug_add in aug_list:
                if not os.path.exists(os.path.join(aug_add, category, sub_category)):
                    os.makedirs(os.path.join(aug_add, category, sub_category))
            command = f"cp -r {os.path.join(input_add, category, sub_category)} {os.path.join(out_add_img, category)}"
            output = subprocess.run(command, shell=True, capture_output=True, text=True)
            if not os.path.exists(os.path.join(out_add_msk, category)):
                os.makedirs(os.path.join(out_add_msk, category))
            command = f"cp -r {os.path.join(input_add_msk, category, sub_category)} {os.path.join(out_add_msk, category)}"
            output = subprocess.run(command, shell=True, capture_output=True, text=True)
            if not sub_category.endswith(".json"): # keep it
                data_aug = ['']
                image_out = os.path.join(out_add_img, category, sub_category) 
                png_files = [f for f in os.listdir(image_out) if f.endswith('.png')]
                for image_name in png_files:
                    img = cv2.imread(os.path.join(out_add_img, category, sub_category, image_name))
                    img_bri = adjust_brightness(img, brightness=-30) #-50 to 50
                    img_con = adjust_contrast(img, contrast=0.9) #0.8,1.5
                    img_gam = adjust_gamma(img, gamma=0.9) # 0.8 to 1.2 
                    cv2.imwrite(os.path.join(out_add_img_bri1, category, sub_category, image_name), img_bri)
                    cv2.imwrite(os.path.join(out_add_img_con1, category, sub_category, image_name), img_con)
                    cv2.imwrite(os.path.join(out_add_img_gam1, category, sub_category, image_name), img_gam)
                    img_bri = adjust_brightness(img, brightness=30)
                    img_con = adjust_contrast(img, contrast=1.4)
                    img_gam = adjust_gamma(img, gamma=1.1)
                    cv2.imwrite(os.path.join(out_add_img_bri2, category, sub_category, image_name), img_bri)
                    cv2.imwrite(os.path.join(out_add_img_con2, category, sub_category, image_name), img_con)
                    cv2.imwrite(os.path.join(out_add_img_gam2, category, sub_category, image_name), img_gam)

def run_colmap(prefix, out_add1, category_list, sub_category_list):
    pref_add_img = os.path.join(prefix, "final_data_2024")
    pref_add_img_bri1 = pref_add_img + "_bri1"
    pref_add_img_con1 = pref_add_img + "_con1"
    pref_add_img_gam1 = pref_add_img + "_gam1"
    pref_add_img_bri2 = pref_add_img + "_bri2"
    pref_add_img_con2 = pref_add_img + "_con2"
    pref_add_img_gam2 = pref_add_img + "_gam2"
    pref_mask = os.path.join(prefix, "final_data_2024_mask")
    aug_list = [pref_add_img, pref_add_img_bri1, pref_add_img_con1, pref_add_img_gam1, \
        pref_add_img_bri2, pref_add_img_con2, pref_add_img_gam2]
    aug_name_list = ['img', 'bri1', 'con1', 'gam1', 'bri2', 'con2', 'gam2']
    for num, pref_add in enumerate(aug_list):
        for category in category_list:
            category_dir = os.path.join(pref_add, category)
            for sub_category in sub_category_list:
                if not sub_category.endswith(".json"):
                    image_add = os.path.join(category_dir, sub_category)
                    mask_add = os.path.join(pref_mask, category, sub_category)
                    image_out1 = os.path.join(out_add1+aug_name_list[num], category, sub_category)
                    if not os.path.exists(image_out1):
                        os.makedirs(image_out1)
                    command = f"bash run_colmap_impr.sh {image_add} {mask_add} {image_out1}"
                    output = subprocess.run(command, shell=True, capture_output=True, text=True)
                    print(output.stdout)
                    command = f"rm -rf {image_out1}/dense/stereo"
                    output = subprocess.run(command, shell=True, capture_output=True, text=True)
                    print(output.stdout)

def load_point_cloud(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    return pcd

def compute_hull_metrics(pcd):
    points = np.asarray(pcd.points)
    hull = ConvexHull(points)
    volume = hull.volume
    surface_area = hull.area
    return volume, surface_area

def compute_convexity(pcd):
    volume_hull, surface_area = compute_hull_metrics(pcd)
    bounding_box_volume = compute_bounding_box_metrics(pcd)
    convexity = volume_hull / bounding_box_volume
    return convexity

def compute_bounding_box_metrics(pcd):
    points = np.asarray(pcd.points)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    bounding_box_volume = np.prod(max_bound - min_bound)
    return bounding_box_volume

def evaluate_point_cloud(ply_file, density_radius=0.05):
    pcd = load_point_cloud(ply_file)
    volume, surface_area = compute_hull_metrics(pcd)
    compactness = volume / (surface_area ** 2)
    print(f"Compactness: {compactness}")
    convexity = compute_convexity(pcd)
    print(f"Convexity: {convexity}")

def main_evaluate(out_add1):
    aug_name_list = ['img', 'bri1', 'con1', 'gam1', 'bri2', 'con2', 'gam2']
    for aug_name in aug_name_list:
        address = out_add1 + aug_name
        category_list = ["rubberduck"]
        for category in category_list:
            category_dir = os.path.join(address, category)
            sub_category_list = ['before_missing']
            for sub_category in sub_category_list:
                sub_cate = os.path.join(category_dir, sub_category)
                point_add = os.path.join(sub_cate, 'dense', 'result.ply')
                print('--------------------', aug_name, '--------------------')
                evaluate_point_cloud(point_add)
                point_add = os.path.join(sub_cate, 'dense', 'images')
                png_files = [f for f in os.listdir(point_add) if f.endswith('.png')]
                png_count = len(png_files)
                print('Length :', png_count)

def undistort_img(pref_add, out_add, camera_add, categories_list):
    pref_add =   os.path.join(pref_add, "final_data_2024")
    pref_mask =  os.path.join(pref_add, "final_data_2024_mask")
    for category in categories_list:
        category_dir = os.path.join(pref_add, category)
        sub_category_list = [d for d in os.listdir(category_dir)]
        for sub_category in sub_category_list:
            if sub_category != "camera" and not sub_category.endswith("json"):
                image_add = os.path.join(pref_add, category, sub_category)
                mask_add = os.path.join(pref_mask, category, sub_category)
                image_out1 = os.path.join(out_add, category, sub_category)
                if not os.path.exists(image_out1):
                    os.makedirs(image_out1)
                command = f"bash run_colmap_impr2.sh {image_add} {mask_add} {camera_add} {image_out1}"
                print("~~~~~~~~Input: ", command)
                output = subprocess.run(command, shell=True, capture_output=True, text=True)
                print(output.stdout)

if __name__ == "__main__":
    step = 4
    if step == 1:
        # ------------ Place for preprocess
        input_add = "example"
        out_add = "example_pre"
        category_list = ["rubberduck"] # Category
        sub_category_list = ['before_missing'] # SubCategory
        create_preprocess(input_add, out_add, category_list, sub_category_list)
    if step == 2:
        # ------------ Place for preprocess
        prefix = "example_pre"
        out_add1 = "example_col_list/colmap"
        category_list = ["rubberduck"] # Category
        sub_category_list = ['before_missing'] # SubCategory
        run_colmap(prefix, out_add1, category_list, sub_category_list)
    if step == 3:
        # ------------ Place for evaluating result
        out_add1 = "example_col_list/colmap"
        main_evaluate(out_add1)
    if step == 4:
        # ------------ Using selected camera information to undistort image
        input_add = "example"
        out_add2 =    "example_refine"
        camera_add =  "/root/autodl-tmp/kai/code/4_anomaly/RAD/example_col_list/colmapcon2/rubberduck/before_missing/camera"
        category_list = ["rubberduck"]
        undistort_img(input_add, out_add2, camera_add, category_list)
    

    