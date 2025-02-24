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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_search.log"),
        logging.StreamHandler()
    ]
)


class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None, masks_output_dir=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.masks_output_dir = masks_output_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

        if masks_output_dir and not os.path.exists(masks_output_dir):
            os.makedirs(masks_output_dir)

        # Создание category_map из первой аннотации
        first_ann_path = os.path.join(annotations_dir, os.listdir(annotations_dir)[0])
        with open(first_ann_path, 'r') as f:
            annotation = json.load(f)
        self.category_map = {category['id']: category['name'] for category in annotation['categories']}

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
                masks.append(mask)
            elif isinstance(ann['segmentation'], list):
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                segmentation = np.array(ann['segmentation'])
                polygon = np.array(segmentation, dtype=np.int32)
                mask = cv2.fillPoly(mask, [polygon], 1)
                masks.append(mask)

        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

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

        return img, target


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)


def calculate_iou(box1, box2):

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Площадь пересечения
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Площади каждого бокса
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def compute_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truths)

    matched_gt = set()

    for pred in predictions:
        pred_box = pred["bbox"]
        pred_image_id = pred["image_id"]
        pred_category_id = pred["category_id"]

        best_iou = 0
        best_gt_idx = None

        for idx, gt in enumerate(ground_truths):
            if (
                gt["image_id"] == pred_image_id
                and gt["category_id"] == pred_category_id
                and idx not in matched_gt
            ):
                gt_box = gt["bbox"]
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

        if best_iou >= iou_threshold:
            true_positives += 1
            false_negatives -= 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    return precision, recall


def compute_map(predictions, ground_truths, iou_threshold=0.5):
    predictions.sort(key=lambda x: x["score"], reverse=True)

    precisions = []
    recalls = []

    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truths)

    matched_gt = set()

    for pred in predictions:
        pred_box = pred["bbox"]
        pred_image_id = pred["image_id"]
        pred_category_id = pred["category_id"]

        best_iou = 0
        best_gt_idx = None

        # Поиск лучшего соответствия среди ground truth
        for idx, gt in enumerate(ground_truths):
            if (
                gt["image_id"] == pred_image_id
                and gt["category_id"] == pred_category_id
                and idx not in matched_gt
            ):
                gt_box = gt["bbox"]
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

        if best_iou >= iou_threshold:
            true_positives += 1
            false_negatives -= 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1

        # Вычисление текущих Precision и Recall
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)

        precisions.append(precision)
        recalls.append(recall)

    # Интерполяция Precision-Recall кривой
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    unique_recalls = np.unique(recalls)
    avg_precisions = []

    for r in unique_recalls:
        # Максимальное Precision для данного Recall
        max_precision = np.max(precisions[recalls >= r])
        avg_precisions.append(max_precision)

    map_score = np.mean(avg_precisions)
    return map_score


def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()

    predictions = []
    ground_truths = []

    try:
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Evaluating", unit='batch'):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                for i, output in enumerate(outputs):
                    image_id = targets[i]["image_id"].item()
                    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                        box = box.cpu().numpy()
                        x_min, y_min, x_max, y_max = box
                        w = x_max - x_min
                        h = y_max - y_min

                        predictions.append({
                            "image_id": image_id,
                            "category_id": label.item(),
                            "bbox": [x_min, y_min, w, h],
                            "score": score.item(),
                        })

                    gt_boxes = targets[i]["boxes"].cpu().numpy()
                    gt_labels = targets[i]["labels"].cpu().numpy()
                    for box, label in zip(gt_boxes, gt_labels):
                        x_min, y_min, x_max, y_max = box
                        w = x_max - x_min
                        h = y_max - y_min

                        ground_truths.append({
                            "image_id": image_id,
                            "category_id": label.item(),
                            "bbox": [x_min, y_min, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        })

        precision, recall = compute_precision_recall(predictions, ground_truths, iou_threshold)
        map_score = compute_map(predictions, ground_truths, iou_threshold)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        logging.info(f"Evaluation results: mAP={map_score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1_score:.4f}")

        return {
            "mAP": map_score,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score
        }

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        return {
            "mAP": 111,
            "Precision": 111,
            "Recall": 111,
            "F1-score": 111
        }


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [32, 48, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    num_epochs = trial.suggest_int("num_epochs", 50, 100)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
    use_augmentation = trial.suggest_categorical("use_augmentation", [True, False])
    model_version = trial.suggest_categorical("model_version", ["v1", "v2"])

    logging.info(f"Trial {trial.number}: Parameters - "
                 f"batch_size={batch_size}, "
                 f"learning_rate={learning_rate:.2e}, "
                 f"momentum={momentum:.2f}, "
                 f"weight_decay={weight_decay:.2e}, "
                 f"num_epochs={num_epochs}, "
                 f"optimizer={optimizer_name}, "
                 f"scheduler={scheduler_name}, "
                 f"use_augmentation={use_augmentation}, "
                 f"model_version={model_version}")

    if model_version == "v1":
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_version == "v2":
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if use_augmentation:
        transform = T.Compose([
            T.RandomHorizontalFlip(p=trial.suggest_float("flip_prob", 0.0, 0.5)),
            T.RandomRotation(degrees=trial.suggest_int("rotation_degrees", 0, 30)),
            T.ColorJitter(brightness=trial.suggest_float("brightness", 0.0, 0.5)),
            T.ToTensor()
        ])
    else:
        transform = T.Compose([T.ToTensor()])

    train_dataset = CustomDataset(
        images_dir='dataset_for_search_1/train/images',
        annotations_dir='dataset_for_search_1/train/annotations',
        transforms=transform
    )
    val_dataset = CustomDataset(
        images_dir='dataset_for_search_1/val/images',
        annotations_dir='dataset_for_search_1/val/annotations',
        transforms=T.ToTensor()
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

    for images, targets in train_loader:
        for t in targets:
            assert "boxes" in t and "labels" in t and "image_id" in t, "Missing required fields in targets"
            assert t["boxes"].shape[1] == 4, "Boxes must have shape [N, 4]"
            assert t["labels"].ndim == 1, "Labels must be 1D tensor"
        break

    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "SGD":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    if scheduler_name == "StepLR":
        scheduler = StepLR(optimizer, step_size=trial.suggest_int("step_size", 5, 20), gamma=0.1)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1)

    # Обучение модели
    best_f1_score = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Обучение на тренировочном наборе
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()

        # Вычисление метрик на валидационном наборе
        try:
            val_metrics = evaluate_model(model, val_loader, device)
            f1_score = val_metrics['F1-score']
            precision = val_metrics['Precision']
            recall = val_metrics['Recall']
            mAP50 = val_metrics['mAP']

            logging.info(f"Trial {trial.number} - Epoch {epoch + 1}/{num_epochs}: "
                         f"F1-score = {f1_score:.4f}, "
                         f"Precision = {precision:.4f}, "
                         f"Recall = {recall:.4f}, "
                         f"mAP50 = {mAP50:.4f}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            f1_score = 0.0

        # Сохранение лучшей модели
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            torch.save(model.state_dict(), f"best_model_trial_{trial.number}.pth")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(f1_score)
        else:
            scheduler.step()

    return best_f1_score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

