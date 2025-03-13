import logging
import os
import json
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
from pycocotools import mask as coco_mask


log_file = "data_check.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


def load_mask_and_convert_to_polygons(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            contour = contour.flatten().tolist()
            polygons.append(contour)
    return polygons


def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [polygon], 1)
    return mask


class CustomDataset:
    def __init__(self, images_dir, annotations_dir, processor=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        ann_path = os.path.join(annotations_dir, self.image_files[0].replace('.jpg', '.json'))
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        self.category_map = {category['id']: category['name'] for category in annotation['categories']}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        FIXED_SIZE = (528, 528)

        img_filename = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_filename)
        img = Image.open(img_path).convert("RGB")

        # Преобразуем изображение к фиксированному размеру
        img = img.resize(FIXED_SIZE, Image.Resampling.LANCZOS)

        ann_filename = img_filename.replace('.jpg', '.json')
        ann_path = os.path.join(self.annotations_dir, ann_filename)
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        image_info = annotation["images"][0]
        boxes = []
        labels = []
        masks = []
        segmentation_masks = []

        for ann in annotation["annotations"]:
            xmin, ymin, width, height = ann["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann["category_id"])

            if isinstance(ann["segmentation"], str):
                mask_path = ann["segmentation"]
                if not os.path.exists(mask_path):
                    logging.error(f"Mask file not found: {mask_path}")
                    continue
                polygons = load_mask_and_convert_to_polygons(mask_path)
                segmentation_masks.append(polygons)
                mask = polygons_to_mask(polygons, FIXED_SIZE[0], FIXED_SIZE[1])  # Используем фиксированный размер
                masks.append(mask)
            elif isinstance(ann["segmentation"], list):
                polygons = ann["segmentation"]
                segmentation_masks.append(polygons)
                mask = polygons_to_mask(polygons, FIXED_SIZE[0], FIXED_SIZE[1])  # Используем фиксированный размер
                masks.append(mask)

        coco_annotations = {
            "image_id": image_info["id"],
            "annotations": [
                {
                    "bbox": [xmin, ymin, width, height],
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": width * height,
                    "iscrowd": 0
                }
                for xmin, ymin, width, height, category_id, segmentation in zip(
                    [b[0] for b in boxes],
                    [b[1] for b in boxes],
                    [b[2] - b[0] for b in boxes],
                    [b[3] - b[1] for b in boxes],
                    labels,
                    segmentation_masks
                )
            ],
        }

        if self.processor:
            encoding = self.processor(
                images=img,
                annotations=coco_annotations,
                return_tensors="pt",
                return_segmentation_masks=True
            )
            pixel_values = encoding["pixel_values"].squeeze()
            if "labels" in encoding and len(encoding["labels"]) > 0:
                label_data = encoding["labels"][0]
                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "class_labels": torch.tensor(labels, dtype=torch.int64),
                    "masks": torch.tensor(np.array(masks), dtype=torch.bool),
                }
            else:
                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "class_labels": torch.tensor(labels, dtype=torch.int64),
                    "masks": torch.tensor(np.array(masks), dtype=torch.bool),
                }
        else:
            pixel_values = img
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "class_labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.tensor(np.array(masks), dtype=torch.bool),
            }

        return pixel_values, target, coco_annotations


def visualize_sample(image, target):
    fig, ax = plt.subplots(1)

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    ax.imshow(image)

    boxes = target["boxes"]
    labels = target["class_labels"]

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box

        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"Class {label}", color='white', fontsize=10, backgroundcolor='red')

    masks = target["masks"]
    if isinstance(masks, torch.Tensor):
        masks = masks.numpy()

    for i, mask in enumerate(masks):
        mask = mask.astype(np.float32)
        mask = np.ma.masked_where(mask == 0, mask)
        ax.imshow(mask, alpha=0.5, cmap='viridis')

    plt.show()


def check_data():
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        do_normalize=False,
        do_resize=False,
        do_rescale=False
    )

    train_dataset = CustomDataset(
        images_dir="dataset_coco_neuro_1/train/images",
        annotations_dir="dataset_coco_neuro_1/train/annotations",
        processor=processor
    )

    for i in range(min(3, len(train_dataset))):
        img, target, coco_annotations = train_dataset[i]

        logging.info(f"Sample {i}:")
        logging.info(f"Image shape: {img.shape}")
        logging.info(f"Target keys: {target.keys()}")
        logging.info(f"Boxes shape: {target['boxes'].shape}")
        logging.info(f"Labels shape: {target['class_labels'].shape}")
        if "masks" in target:
            logging.info(f"Masks shape: {target['masks'].shape}")

        visualize_sample(img, target)

        try:
            encoding = processor(
                images=img,
                annotations=coco_annotations,
                return_tensors="pt",
                return_segmentation_masks=True
            )
            logging.info("Data is compatible with DetrImageProcessor.")
        except Exception as e:
            logging.error(f"Error processing data: {e}")


if __name__ == "__main__":
    check_data()
