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
    logging.FileHandler("training_screen_mask_rcnn.log"),
    logging.StreamHandler()
])


def calculate_metrics(model, data_loader, device):
    num_classes = 3

    metric_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='macro')
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='macro')
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    model.eval()

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            preds_map = []
            targets_map = []

            for pred, target in zip(predictions, targets):
                preds_map.append({
                    "boxes": pred["boxes"].cpu(),
                    "scores": pred["scores"].cpu(),
                    "labels": pred["labels"].cpu(),
                })
                targets_map.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                })

                iou_matrix = box_iou(pred["boxes"], target["boxes"])
                max_iou, matched_gt_indices = torch.max(iou_matrix, dim=1)

                valid_preds = max_iou >= 0.5
                matched_pred_labels = pred["labels"][valid_preds].cpu()
                matched_gt_labels = target["labels"][matched_gt_indices[valid_preds]].cpu()

                if len(matched_pred_labels) > 0 and len(matched_gt_labels) > 0:
                    precision_metric.update(matched_pred_labels, matched_gt_labels)
                    recall_metric.update(matched_pred_labels, matched_gt_labels)
                    f1_metric.update(matched_pred_labels, matched_gt_labels)

            metric_map.update(preds_map, targets_map)

    result_map = metric_map.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()

    return {
        "map_50": result_map["map_50"].item() if "map_50" in result_map else 0,
        "map_50_95": result_map["map"].item() if "map" in result_map else 0,
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item()
    }


model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

num_classes = 3

in_features = model.roi_heads.box_predictor.cls_score.in_features
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = CustomDataset(
    images_dir='../dataset_coco_neuro_1/train/images',
    annotations_dir='../dataset_coco_neuro_1/train/annotations',
    transforms=transform
)

val_dataset = CustomDataset(
    images_dir='../dataset_coco_neuro_1/val/images',
    annotations_dir='../dataset_coco_neuro_1/val/annotations',
    transforms=torchvision.transforms.ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=3.63e-04, weight_decay=3.16e-04)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)


# Тренировка модели
with open('maskRCNN_result_screen.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time', 'Train/box_loss', 'Train/cls_loss', 'Train/dfl_loss',
                     'Metrics/Precision', 'Metrics/Recall', 'Metrics/F1', 'Metrics/mAP50', 'Metrics/mAP50-95',
                     'Val/box_loss', 'Val/cls_loss', 'Val/dfl_loss', 'LR/pg0'])

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()

        # Вычисление метрик на валидации
        val_metrics = calculate_metrics(model, val_loader, device)
        f1_score = val_metrics['f1']

        logging.info(f"Epoch {epoch + 1}/{num_epochs}: "
                     f"F1-score = {f1_score:.4f}, "
                     f"Precision = {val_metrics['precision']:.4f}, "
                     f"Recall = {val_metrics['recall']:.4f}, "
                     f"mAP50 = {val_metrics['map_50']:.4f}")

        torch.save(model.state_dict(), f"model_mask_rcnn_screen.pth")

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(f1_score)
        else:
            scheduler.step()


torch.save(model.state_dict(), 'mask_rcnn_model_screen.pth')


