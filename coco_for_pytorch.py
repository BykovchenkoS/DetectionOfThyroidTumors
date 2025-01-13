import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

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
        category_map = {category['id']: category['name'] for category in annotation['categories']}

        img_width, img_height = img.size

        for ann in annotation['annotations']:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
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
# print("Image shape:", img.shape)
# print("Annotations:", target)


def visualize(img, target, category_map):
    img = img.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    boxes = target['boxes']
    labels = target['labels']
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        category_name = category_map[label.item()]
        ax.text(xmin, ymin - 5, category_name, color='r', fontsize=10, fontweight='bold')

    plt.show()


def visualize_batch(images, targets, category_map):
    batch_size = len(images)
    fig, axs = plt.subplots(1, batch_size, figsize=(15, 5))

    if batch_size == 1:
        axs = [axs]  # Ensure axs is iterable even for batch size 1

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

        ax.axis('off')  # Hide axis for better visualization

    plt.tight_layout()
    plt.show()


# visualize(img, target, category_map)
