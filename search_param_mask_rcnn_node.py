import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights
import optuna
from torch.optim.lr_scheduler import StepLR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
from PIL import Image
import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from coco_for_pytorch import CustomDataset
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import Precision, Recall, F1Score


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_mask_r_cnn_node.log"),
        logging.StreamHandler()
    ]
)


def calculate_metrics(model, data_loader, device):
    num_classes = 2

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


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    num_epochs = trial.suggest_int("num_epochs", 20, 100)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
    model_version = trial.suggest_categorical("model_version", ["v1", "v2"])

    logging.info(f"Trial {trial.number}: Parameters - "
                 f"batch_size={batch_size}, "
                 f"learning_rate={learning_rate:.2e}, "
                 f"momentum={momentum:.2f}, "
                 f"weight_decay={weight_decay:.2e}, "
                 f"num_epochs={num_epochs}, "
                 f"optimizer={optimizer_name}, "
                 f"scheduler={scheduler_name}, "
                 f"model_version={model_version}")

    if model_version == "v1":
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_version == "v2":
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
        images_dir='dataset_for_search_2/train/images',
        annotations_dir='dataset_for_search_2/train/annotations',
        transforms=transform
    )
    val_dataset = CustomDataset(
        images_dir='dataset_for_search_2/val/images',
        annotations_dir='dataset_for_search_2/val/annotations',
        transforms=torchvision.transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Оптимизатор
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "SGD":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # Планировщик скорости обучения
    if scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 5, 20), gamma=0.1)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1)

    # Обучение модели
    best_f1_score = 0.0
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

        logging.info(f"Trial {trial.number} - Epoch {epoch + 1}/{num_epochs}: "
                     f"F1-score = {f1_score:.4f}, "
                     f"Precision = {val_metrics['precision']:.4f}, "
                     f"Recall = {val_metrics['recall']:.4f}, "
                     f"mAP50 = {val_metrics['map_50']:.4f}")

        # Сохранение лучшей модели
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            torch.save(model.state_dict(), f"best_model_trial_{trial.number}_node.pth")

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(f1_score)
        else:
            scheduler.step()

    return best_f1_score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

