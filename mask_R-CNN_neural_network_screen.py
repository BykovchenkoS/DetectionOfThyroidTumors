import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
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
        }
        assert min(labels) >= 1, "Labels should start from 1"
        assert max(labels) <= len(category_map), "Labels exceed the number of classes"

        if self.transforms:
            img = self.transforms(img)

        return img, target


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("training_screen.log"),
    logging.StreamHandler()
])

model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

num_classes = 3

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

transform = T.Compose([T.ToTensor()])

dataset = CustomDataset(images_dir='dataset_for_search_1/train/images',
                        annotations_dir='dataset_for_search_1/train/annotations',
                        transforms=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


val_dataset = CustomDataset(
    images_dir='dataset_for_search_1/val/images',
    annotations_dir='dataset_for_search_1/val/annotations',
    transforms=transform
)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


# Оптимизатор
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


def calculate_metrics(model, data_loader, device):
    # Инициализация метрики MeanAveragePrecision
    metric_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    # Инициализация метрик Precision, Recall, F1Score
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
                # Для MeanAveragePrecision
                preds_map.append({
                    "boxes": pred["boxes"].cpu(),
                    "scores": pred["scores"].cpu(),
                    "labels": pred["labels"].cpu(),
                })
                targets_map.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                })

                # Для Precision, Recall, F1Score
                # Выполняем матчинг между предсказаниями и ground truth
                iou_matrix = box_iou(pred["boxes"], target["boxes"])
                max_iou, matched_gt_indices = torch.max(iou_matrix, dim=1)

                # Фильтруем предсказания с IoU >= порога
                valid_preds = max_iou >= 0.5
                matched_pred_labels = pred["labels"][valid_preds].cpu()
                matched_gt_labels = target["labels"][matched_gt_indices[valid_preds]].cpu()

                # Обновляем метрики Precision, Recall, F1Score
                if len(matched_pred_labels) > 0 and len(matched_gt_labels) > 0:
                    precision_metric.update(matched_pred_labels, matched_gt_labels)
                    recall_metric.update(matched_pred_labels, matched_gt_labels)
                    f1_metric.update(matched_pred_labels, matched_gt_labels)

            # Обновляем метрику MeanAveragePrecision
            metric_map.update(preds_map, targets_map)

    # Рассчитываем финальные значения метрик
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

# Тренировка модели
with open('maskRCNN_result_screen.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time', 'Train/box_loss', 'Train/cls_loss', 'Train/dfl_loss',
                     'Metrics/Precision', 'Metrics/Recall', 'Metrics/F1', 'Metrics/mAP50', 'Metrics/mAP50-95',
                     'Val/box_loss', 'Val/cls_loss', 'Val/dfl_loss', 'LR/pg0'])

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        for images, targets in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)

            box_loss = loss_dict['loss_box_reg'].item()
            cls_loss = loss_dict['loss_classifier'].item()
            dfl_loss = loss_dict['loss_objectness'].item()

            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            losses.backward()
            optimizer.step()

        epoch_time = time.time() - start_time

        # Вычисление метрик на валидации
        val_metrics = calculate_metrics(model, val_loader, device)

        writer.writerow([epoch + 1, epoch_time, box_loss, cls_loss, dfl_loss,
                         val_metrics["precision"], val_metrics["recall"], val_metrics["f1"],
                         val_metrics["map_50"], val_metrics["map_50_95"],
                         'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
                         optimizer.param_groups[0]['lr']])
        file.flush()

        log_message = (f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(data_loader)} - "
                       f"Box Loss: {box_loss} - Class Loss: {cls_loss} - Objectness Loss: {dfl_loss}\n"
                       f"Validation - Precision: {val_metrics['precision']:.4f}, "
                       f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, "
                       f"mAP50: {val_metrics['map_50']:.4f}, mAP50-95: {val_metrics['map_50_95']:.4f}")
        logging.info(log_message)

        print(log_message)

torch.save(model.state_dict(), 'mask_rcnn_model_screen.pth')


