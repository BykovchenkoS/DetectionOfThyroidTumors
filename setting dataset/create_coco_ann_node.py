import json
import os
import numpy as np
from PIL import Image
import cv2

ann_folder = 'dataset_coco_neuro_2/shifted_json'
ann_new_folder = 'ann_new_2_node/'
masks_folder = 'screen foto/dataset 2024-04-21 14_33_36/old_masks/'
images_folder = 'dataset_coco_neuro_2/images_neuro_2/'


def convert_to_coco_format(annotation, image_filename):
    image_path = os.path.join(images_folder, image_filename)
    image_width = annotation['size']['width']
    image_height = annotation['size']['height']
    annotations = []

    categories = [
        {"id": 0, "name": "Node", "supercategory": "organ"}
    ]

    for obj in annotation['objects']:
        class_title = obj['classTitle'].strip()

        if class_title == "Node":
            if 'bitmap' in obj:
                mask_filename = f"{os.path.splitext(image_filename)[0]}_{class_title.replace(' ', '_')}_{obj['id']}.png"
                mask_path = os.path.join(masks_folder, mask_filename)

                if os.path.exists(mask_path):
                    origin_x = obj['bitmap']['origin'][0]
                    origin_y = obj['bitmap']['origin'][1]
                    coco_bbox = calculate_bbox_from_mask(mask_path, origin_x, origin_y)
                    area = int(np.sum(coco_bbox))

                    annotations.append({
                        "id": len(annotations),
                        "image_id": int(os.path.splitext(image_filename)[0]),
                        "category_id": 0,
                        "segmentation": mask_path,
                        "area": area,
                        "bbox": coco_bbox,
                        "iscrowd": 0
                    })
                else:
                    print(f"Warning: Mask {mask_filename} not found.")
            elif 'points' in obj:
                points = obj['points']['exterior']
                segmentation = points
                area = int(calculate_area(segmentation))
                bbox = calculate_bbox_from_points(segmentation)
                annotations.append({
                    "id": len(annotations),
                    "image_id": int(os.path.splitext(image_filename)[0]),
                    "category_id": 0,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": [int(coord) for coord in bbox],
                    "iscrowd": 0
                })

    return {
        "images": [{"id": int(os.path.splitext(image_filename)[0]), "file_name": image_path, "width": image_width, "height": image_height}],
        "annotations": annotations,
        "categories": categories
    }


def calculate_area(points):
    area = 0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area


def calculate_bbox_from_points(points):
    x_min = min(p[0] for p in points)
    y_min = min(p[1] for p in points)
    x_max = max(p[0] for p in points)
    y_max = max(p[1] for p in points)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def calculate_bbox_from_mask(mask_path, origin_x, origin_y):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    indices = np.where(mask != 0)
    y_min, y_max = np.min(indices[0]), np.max(indices[0])
    x_min, x_max = np.min(indices[1]), np.max(indices[1])
    bbox = [origin_x, origin_y, x_max - x_min, y_max - y_min]
    return bbox


def convert_numpy_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_numpy_to_native(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    else:
        return obj


def create_new_annotations():
    if not os.path.exists(ann_new_folder):
        os.makedirs(ann_new_folder)

    for filename in os.listdir(ann_folder):
        if filename.endswith('.json'):
            annotation_path = os.path.join(ann_folder, filename)
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)

            image_filename = f"{os.path.splitext(filename)[0]}.jpg"
            coco_annotation = convert_to_coco_format(annotation, image_filename)
            coco_annotation = convert_numpy_to_native(coco_annotation)

            new_annotation_path = os.path.join(ann_new_folder, filename)
            with open(new_annotation_path, 'w', encoding='utf-8') as f:
                json.dump(coco_annotation, f, indent=4)

            print(f"Аннотация для {filename} успешно обновлена.")


create_new_annotations()
