import json
import cv2
import numpy as np
import os


def resize_mask_from_annotations(image_path, mask_path, annotation, image_data, output_dir):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file at {image_path} was not found.")

    img_height = image_data["height"]
    img_width = image_data["width"]

    masks = []

    if isinstance(mask_path, list):
        print(f"Skipping polygon mask for annotation {annotation['id']}")
        return
    elif isinstance(mask_path, str) and mask_path.endswith(".png"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file at {mask_path} not found.")
        masks.append(mask)
    elif isinstance(mask_path, dict) and "counts" in mask_path:
        rle_mask = decode_rle(mask_path, img_height, img_width)
        masks.append(rle_mask)
    else:
        raise ValueError("Unsupported type for segmentation.")

    x, y, width, height = annotation["bbox"]
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        new_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        new_mask[int(y):int(y) + int(height), int(x):int(x) + int(width)] = mask_resized

        if isinstance(mask_path, str):
            output_mask_filename = f"{output_dir}/{os.path.basename(mask_path)}"
            cv2.imwrite(output_mask_filename, new_mask)
            print(f"New mask saved at {output_mask_filename}")


def decode_rle(rle, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    counts = list(map(int, rle["counts"].split()))
    for i in range(0, len(counts), 2):
        start = counts[i]
        length = counts[i + 1]
        mask.flat[start:start + length] = 255
    return mask


def load_annotations(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)


json_folder_path = 'dataset_coco_neuro_1/train/annotations'
output_dir = 'new_mask'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all JSON files in the directory
json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

try:
    for json_file in json_files:
        json_file_path = os.path.join(json_folder_path, json_file)
        json_data = load_annotations(json_file_path)
        annotations = json_data["annotations"]
        image_data = json_data["images"][0]

        image_id = image_data["id"]
        image_path = f"dataset_coco_neuro_1/train/images/{image_id}.jpg"

        # Process all annotations in the current JSON file
        for i, annotation in enumerate(annotations):
            mask_path = annotation["segmentation"]
            resize_mask_from_annotations(image_path, mask_path, annotation, image_data, output_dir)

except FileNotFoundError as e:
    print(e)
except json.JSONDecodeError:
    print("Error decoding JSON file.")
except ValueError as e:
    print(e)
