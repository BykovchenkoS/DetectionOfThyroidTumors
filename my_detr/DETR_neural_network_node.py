import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import cv2
from detr.engine import train_one_epoch, evaluate
import detr.util.misc as utils
from detr.models.detr import build as build_model
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
from detr.models.detr import PostProcess, PostProcessSegm
from pycocotools.coco import COCO
import tempfile
import json
import logging
import os
import sys


class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None, masks_output_dir=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.masks_output_dir = masks_output_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        if masks_output_dir and not os.path.exists(masks_output_dir):
            os.makedirs(masks_output_dir)
        first_ann_path = os.path.join(annotations_dir, self.image_files[0].replace('.jpg', '.json'))
        with open(first_ann_path, 'r') as f:
            annotation = json.load(f)
        self.category_map = {category['id']: category['name'] for category in annotation['categories']}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_filename = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_filename)
        img = Image.open(img_path).convert("RGB")

        img_width, img_height = img.size
        orig_size = torch.tensor([img_height, img_width])

        ann_filename = img_filename.replace('.jpg', '.json')
        ann_path = os.path.join(self.annotations_dir, ann_filename)
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        boxes = []
        labels = []
        masks = []
        img_width, img_height = img.size
        for ann in annotation['annotations']:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])
            if isinstance(ann['segmentation'], str):
                mask_path = ann['segmentation']
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask)
                mask = (mask > 0).astype(np.uint8)
                mask = Image.fromarray(mask)
                masks.append(np.array(mask))
            elif isinstance(ann['segmentation'], list):
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                segmentation = np.array(ann['segmentation'])
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
            "orig_size": orig_size,
            "size": torch.tensor([528, 528])
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def getImgIds(self):
        return [i for i in range(len(self.image_files))]

    def getCatIds(self):
        return list(self.category_map.keys())


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets


def get_model_instance_segmentation(num_classes):
    args = argparse.Namespace()
    args.num_classes = num_classes
    args.pretrained = False
    args.backbone = 'resnet50'
    args.dilation = False
    args.position_embedding = 'sine'
    args.hidden_dim = 256
    args.lr_backbone = 1e-5
    args.masks = True
    args.aux_loss = True
    args.set_cost_class = 1
    args.set_cost_bbox = 5
    args.set_cost_giou = 2
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dataset_file = 'custom'
    args.num_queries = 100
    args.bbox_loss_coef = 5
    args.giou_loss_coef = 2
    args.mask_loss_coef = 1
    args.dice_loss_coef = 1
    args.eos_coef = 0.1
    args.dec_layers = 6
    args.enc_layers = 6
    args.frozen_weights = None
    args.dropout = 0.1
    args.nheads = 8
    args.dim_feedforward = 2048
    args.pre_norm = False
    model, criterion, postprocessors = build_model(args, num_classes)
    return model, criterion, postprocessors


output_file = "detr_training_output_node.log"


class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, "w")

    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


sys.stdout = Logger(output_file)
sys.stderr = sys.stdout


if __name__ == '__main__':
    train_dataset = CustomDataset(
        images_dir='dataset_coco_neuro_2/train/images',
        annotations_dir='dataset_coco_neuro_2/train/annotations',
        transforms=get_transform(train=True)
    )
    train_data_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4,
        collate_fn=custom_collate_fn
    )

    num_classes = len(train_dataset.category_map)
    print("Number of classes:", num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, criterion, postprocessors = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        epoch_metrics = train_one_epoch(model, criterion, train_data_loader, optimizer, device, epoch)

    model_path = "my_detr/detr_node.pth"
    torch.save(model.state_dict(), model_path)
    print("Model has been successfully trained and saved!")

    sys.stdout.file.close()
    sys.stdout = sys.stdout.console
    sys.stderr = sys.__stderr__

