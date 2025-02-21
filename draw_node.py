import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

annotations_file = "dataset_coco_neuro_2/train/annotations/33.json"
with open(annotations_file, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]


def draw_bounding_boxes_and_mask(image_path, mask_path, annotations):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)

    for annotation in annotations:
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]

        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        color = "red" if category_id == 0 else "blue"

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

        class_name = data["categories"][category_id]["name"]
        draw.text((x_min, y_min - 10), class_name, fill=color)

    image_with_mask = np.array(image).astype(np.float32)
    image_with_mask[mask > 0] = [128, 0, 128]
    image_with_mask /= 255.0

    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_mask)
    plt.axis("off")
    plt.show()


for img_info in images:
    image_id = img_info["id"]
    image_path = img_info["file_name"]
    image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]

    for annotation in image_annotations:
        mask_path = annotation["segmentation"]

        print(f"Рисуем bounding box и маску для изображения: {image_path}")
        draw_bounding_boxes_and_mask(image_path, mask_path, [annotation])