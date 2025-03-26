import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as T
from tqdm import tqdm
from coco_for_pytorch import CustomDataset
import csv
import time
import logging
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
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import Precision, Recall, F1Score
from coco_for_pytorch import CustomDataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("training_node_mask_rcnn.log"),
    logging.StreamHandler()
])


def calculate_metrics(model, data_loader, device, confidence_threshold=0.5, iou_threshold=0.5):
    metric_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    precision_metric = Precision(task="binary", average='macro')
    recall_metric = Recall(task="binary", average='macro')
    f1_metric = F1Score(task="binary", average='macro')

    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)

            preds_map = [{"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()} for p in predictions]
            targets_map = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets]
            metric_map.update(preds_map, targets_map)

            for pred, target in zip(predictions, targets):
                valid_preds = pred["scores"] >= confidence_threshold
                pred_boxes = pred["boxes"][valid_preds]
                pred_labels = pred["labels"][valid_preds]

                iou = box_iou(pred_boxes, target["boxes"])
                max_iou, matched_gt_indices = iou.max(dim=1)
                valid_matches = max_iou >= iou_threshold

                if valid_matches.sum() > 0:
                    matched_preds = pred_labels[valid_matches]
                    matched_gts = target["labels"][matched_gt_indices][valid_matches]
                    precision_metric.update(matched_preds, matched_gts)
                    recall_metric.update(matched_preds, matched_gts)
                    f1_metric.update(matched_preds, matched_gts)

    map_results = metric_map.compute()
    return {
        "map_50": map_results["map_50"].item(),
        "map_50_95": map_results["map"].item(),
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
        "f1": f1_metric.compute().item()
    }


model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = CustomDataset(
    images_dir='../dataset_coco_neuro_3/train/images',
    annotations_dir='../dataset_coco_neuro_3/train/annotations',
    transforms=transform
)

val_dataset = CustomDataset(
    images_dir='../dataset_coco_neuro_3/val/images',
    annotations_dir='../dataset_coco_neuro_3/val/annotations',
    transforms=torchvision.transforms.ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)


for images, targets in train_loader:
    print("Images type:", type(images))
    print("Number of images in batch:", len(images))
    print("Image tensor shape:", images[0].shape)

    print("Targets type:", type(targets))
    print("Number of targets in batch:", len(targets))
    for i, target in enumerate(targets):
        print(f"Target {i}:")
        print("  Boxes shape:", target['boxes'].shape)
        print("  Labels shape:", target['labels'].shape)
        if 'masks' in target:
            print("  Masks shape:", target['masks'].shape)

    break


params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=3, verbose=True
)


with open('maskRCNN_result_node.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time', 'Train/box_loss', 'Train/cls_loss', 'Train/mask_loss',
                     'Metrics/Precision', 'Metrics/Recall', 'Metrics/F1', 'Metrics/mAP50', 'Metrics/mAP50-95',
                     'Val/box_loss', 'Val/cls_loss', 'Val/mask_loss', 'LR/pg0'])

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    train_losses = {
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_mask': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0,
        'total_loss': 0.0
    }

    start_time = time.time()

    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()
        losses.backward()
        optimizer.step()

        train_losses['loss_classifier'] += loss_dict['loss_classifier'].item()
        train_losses['loss_box_reg'] += loss_dict['loss_box_reg'].item()
        train_losses['loss_mask'] += loss_dict['loss_mask'].item()
        train_losses['loss_objectness'] += loss_dict['loss_objectness'].item()
        train_losses['loss_rpn_box_reg'] += loss_dict['loss_rpn_box_reg'].item()
        train_losses['total_loss'] += losses.item()

    num_batches = len(train_loader)
    avg_train_losses = {key: value / num_batches for key, value in train_losses.items()}

    epoch_time = time.time() - start_time

    val_metrics = calculate_metrics(model, val_loader, device)
    f1_score = val_metrics['f1']

    logging.info(f"Epoch {epoch + 1}/{num_epochs}: "
                 f"Train Losses - Classifier: {avg_train_losses['loss_classifier']:.4f}, "
                 f"Box Reg: {avg_train_losses['loss_box_reg']:.4f}, "
                 f"Mask: {avg_train_losses['loss_mask']:.4f}, "
                 f"Objectness: {avg_train_losses['loss_objectness']:.4f}, "
                 f"RPN Box Reg: {avg_train_losses['loss_rpn_box_reg']:.4f}, "
                 f"Total: {avg_train_losses['total_loss']:.4f} | "
                 f"Metrics - F1-score: {f1_score:.4f}, "
                 f"Precision: {val_metrics['precision']:.4f}, "
                 f"Recall: {val_metrics['recall']:.4f}, "
                 f"mAP50: {val_metrics['map_50']:.4f}")

    with open('maskRCNN_result_node.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch + 1,
            epoch_time,
            avg_train_losses['loss_box_reg'],
            avg_train_losses['loss_classifier'],
            avg_train_losses['loss_mask'],
            val_metrics['precision'],
            val_metrics['recall'],
            val_metrics['f1'],
            val_metrics['map_50'],
            val_metrics['map_50_95'],
            0,  # Placeholder for Val/box_loss
            0,  # Placeholder for Val/cls_loss
            0,  # Placeholder for Val/mask_loss
            optimizer.param_groups[0]['lr']
        ])

    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(f1_score)
    else:
        scheduler.step()

torch.save(model.state_dict(), 'mask_rcnn_model_node_new_1.pth')


