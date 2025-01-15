import json
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
import torch
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None, masks_output_dir=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.masks_output_dir = masks_output_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

        if masks_output_dir and not os.path.exists(masks_output_dir):
            os.makedirs(masks_output_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_filename = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_filename)
        img = Image.open(img_path).convert("RGB")

        ann_filename = img_filename.replace('.jpg', '.json')
        ann_path = os.path.join(self.annotations_dir, ann_filename)
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        boxes = []
        labels = []
        masks = []
        category_map = {category['id']: category['name'] for category in annotation['categories']}

        img_width, img_height = img.size

        for ann in annotation['annotations']:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])

            if isinstance(ann['segmentation'], str):
                mask_path = ann['segmentation']
                mask = Image.open(mask_path).convert("L")  # Маска в grayscale
                mask = np.array(mask)
                mask = (mask > 0).astype(np.uint8)
                mask = Image.fromarray(mask)
                mask = mask.resize((img_width, img_height), Image.NEAREST)  # Масштабируем маску до размера изображения
                masks.append(np.array(mask))
            elif isinstance(ann['segmentation'], list):  # Полигональная сегментация
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                segmentation = np.array(ann['segmentation'])
                # Заполняем маску для полигона
                polygon = np.array(segmentation, dtype=np.int32)
                mask = cv2.fillPoly(mask, [polygon], 1)
                masks.append(mask)

        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        if self.masks_output_dir:
            for i, mask in enumerate(masks):
                mask_filename = f"{img_filename.replace('.jpg', '')}_mask_{i}.pt"
                mask_path = os.path.join(self.masks_output_dir, mask_filename)
                torch.save(mask, mask_path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([index]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target, category_map


transform = T.Compose([T.ToTensor()])
dataset = CustomDataset(images_dir='dataset_coco_neuro_1/train/images',
                        annotations_dir='dataset_coco_neuro_1/train/annotations',
                        transforms=transform)

img, target, category_map = dataset[0]
print("Image shape:", img.shape)
print("Annotations:", target)


def visualize(img, target, category_map):
    img = img.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    boxes = target['boxes']
    labels = target['labels']
    masks = target['masks']

    for box, label, mask in zip(boxes, labels, masks):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        category_name = category_map[label.item()]
        ax.text(xmin, ymin - 5, category_name, color='r', fontsize=10, fontweight='bold')

        mask_np = mask.numpy()
        contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze(1)
            polygon = patches.Polygon(contour, linewidth=2, edgecolor='blue', fill=False)
            ax.add_patch(polygon)

    plt.show()


def visualize_batch(images, targets, category_map):
    batch_size = len(images)
    fig, axs = plt.subplots(1, batch_size, figsize=(15, 5))

    if batch_size == 1:
        axs = [axs]

    for idx, (img, target) in enumerate(zip(images, targets)):
        img = img.permute(1, 2, 0).numpy()
        ax = axs[idx]
        ax.imshow(img)

        boxes = target['boxes']
        labels = target['labels']
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            category_name = category_map[label.item()]
            ax.text(xmin, ymin - 5, category_name, color='r', fontsize=10, fontweight='bold')

        ax.axis('off')

    plt.tight_layout()
    plt.show()


# visualize(img, target, category_map)