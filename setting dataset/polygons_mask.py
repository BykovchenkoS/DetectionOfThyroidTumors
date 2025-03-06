import json
import cv2
import numpy as np
import os


def resize_mask_from_annotations(image_path, mask_path, annotation, image_data, output_dir, failed_files):
    if not os.path.exists(image_path):
        failed_files.append(image_path)
        print(f"The image file at {image_path} was not found.")
        return

    img_height = image_data["height"]
    img_width = image_data["width"]

    masks = []

    if isinstance(mask_path, list):
        print(f"Skipping polygon mask for annotation {annotation['id']}")
        return
    elif isinstance(mask_path, str) and mask_path.endswith(".png"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            failed_files.append(mask_path)
            print(f"Mask file at {mask_path} not found.")
            return
        masks.append(mask)
    elif isinstance(mask_path, dict) and "counts" in mask_path:
        rle_mask = decode_rle(mask_path, img_height, img_width)
        masks.append(rle_mask)
    else:
        failed_files.append(mask_path)
        raise ValueError("Unsupported type for segmentation.")

    x, y, width, height = annotation["bbox"]
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)

        # Проверка и вывод сообщения в случае несоответствия размеров
        if mask_resized.shape[0] > height or mask_resized.shape[1] > width:
            print(f"Error with mask {mask_path}: "
                  f"Mask size {mask_resized.shape} does not fit in bounding box of size ({height}, {width})")

        # Обрезаем маску, если она выходит за пределы области
        mask_resized = mask_resized[:min(height, mask_resized.shape[0]), :min(width, mask_resized.shape[1])]

        new_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        new_mask[int(y):int(y) + min(height, mask_resized.shape[0]),
        int(x):int(x) + min(width, mask_resized.shape[1])] = mask_resized

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


json_folder_path = '../dataset_coco_neuro_2/val/annotations'
output_dir = 'dataset_coco_neuro_2/new_mask_3'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

failed_files = []

try:
    for json_file in json_files:
        json_file_path = os.path.join(json_folder_path, json_file)
        json_data = load_annotations(json_file_path)
        annotations = json_data["annotations"]
        image_data = json_data["images"][0]

        image_id = image_data["id"]
        image_path = f"dataset_coco_neuro_2/val/images/{image_id}.jpg"

        for i, annotation in enumerate(annotations):
            mask_path = annotation["segmentation"]
            resize_mask_from_annotations(image_path, mask_path, annotation, image_data, output_dir, failed_files)

except FileNotFoundError as e:
    print(e)
except json.JSONDecodeError:
    print("Error decoding JSON file.")
except ValueError as e:
    print(e)

if failed_files:
    print("\nFiles that could not be processed:")
    for failed_file in failed_files:
        print(failed_file)
