import json
import os
import numpy as np
from PIL import Image

# ann_folder = 'screen foto/dataset 2024-04-21 14_33_36/shifted_json'
# ann_new_folder = 'ann_new_1/'
# masks_folder = 'screen foto/dataset 2024-04-21 14_33_36/masks/'
# images_folder = 'screen foto/dataset 2024-04-21 14_33_36/images_neuro_1/'

ann_folder = 'dataset_coco_neuro_1/shifted_json'
ann_new_folder = 'ann_new_2/'
masks_folder = 'dataset_coco_neuro_1/masks/'
images_folder = 'dataset_coco_neuro_1/images_neuro_1/'

def convert_to_coco_format(annotation, image_filename):
    image_path = os.path.join(images_folder, image_filename)
    image_width = annotation['size']['width']
    image_height = annotation['size']['height']
    annotations = []

    categories = [
        {"id": 0, "name": "sagital_longitudinal", "supercategory": "organ"},
        {"id": 1, "name": "Thyroid tissue", "supercategory": "organ"},
        {"id": 2, "name": "Carotis", "supercategory": "organ"}
    ]

    for obj in annotation['objects']:
        class_title = obj['classTitle'].strip()

        if class_title == "sagital":
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

        elif class_title in ["Thyroid tissue", "Carotis"]:
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
                    "category_id": 1 if class_title == "Thyroid tissue" else 2,
                    "segmentation": mask_path,
                    "area": area,
                    "bbox": [int(coord) for coord in coco_bbox],
                    "iscrowd": 0
                })
            else:
                print(f"Warning: Mask {mask_filename} not found.")

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
    mask = Image.open(mask_path).convert('1')
    mask_array = np.array(mask)
    object_pixels = np.where(mask_array == 1)

    xmin = np.min(object_pixels[1]) + origin_x
    xmax = np.max(object_pixels[1]) + origin_x
    ymin = np.min(object_pixels[0]) + origin_y
    ymax = np.max(object_pixels[0]) + origin_y

    bbox = (xmin, ymin, xmax, ymax)
    width = xmax - xmin
    height = ymax - ymin
    coco_bbox = [xmin, ymin, width, height]

    return coco_bbox


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

            new_annotation_path = os.path.join(ann_new_folder, filename)
            with open(new_annotation_path, 'w', encoding='utf-8') as f:
                json.dump(coco_annotation, f, indent=4)

            print(f"Аннотация для {filename} успешно обновлена.")


create_new_annotations()
