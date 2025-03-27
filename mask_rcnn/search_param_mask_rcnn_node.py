import json
import os

import cv2
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# from coco_for_pytorch import CustomDataset
import logging
import torch
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import Precision, Recall, F1Score
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as T

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_mask_rcnn_node.log"),
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


def calculate_metrics(model, data_loader, device):
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

            for pred, target in zip(predictions, targets):
                valid_preds = pred["scores"] > 0.3
                if not valid_preds.any():
                    continue

                pred_boxes = pred["boxes"][valid_preds]
                pred_labels = pred["labels"][valid_preds]

                iou = box_iou(pred_boxes, target["boxes"])
                max_iou, _ = iou.max(dim=1)
                tp = max_iou >= 0.5

                if tp.any() and len(target["labels"]) > 0:
                    precision_metric.update(pred_labels[tp], target["labels"][tp])
                    recall_metric.update(pred_labels[tp], target["labels"][tp])
                    f1_metric.update(pred_labels[tp], target["labels"][tp])

            metric_map.update(
                [{"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]} for p in predictions],
                [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
            )

    result_map = metric_map.compute()
    return {
        "map_50": result_map["map_50"].item() if "map_50" in result_map else 0,
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
        "f1": f1_metric.compute().item()
    }


def visualize_batch(images, targets, max_images=3):
    num_images = min(len(images), max_images)  # Ограничиваем количество изображений
    fig, axes = plt.subplots(1, num_images, figsize=(20, 10))  # Создаем subplot для каждого изображения

    for i in range(num_images):
        img_tensor = images[i]
        target = targets[i]

        img = T.ToPILImage()(img_tensor)  # Преобразование тензора обратно в PIL-изображение
        draw = ImageDraw.Draw(img)

        # Отображение bounding boxes
        for box in target["boxes"]:
            xmin, ymin, xmax, ymax = box.tolist()
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # Отображение масок
        if "masks" in target:
            masks = target["masks"].numpy()
            for mask in masks:
                mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                mask_image = mask_image.convert("RGBA")
                mask_data = np.array(mask_image)
                mask_data[..., -1] = mask * 128  # Прозрачность для маски
                mask_image = Image.fromarray(mask_data)
                img.paste(mask_image, (0, 0), mask_image)

        # Отображение изображения
        if num_images == 1:
            axes.imshow(img)
            axes.axis("off")
            axes.set_title(f"Image {i}")
        else:
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Image {i}")

    plt.tight_layout()
    plt.show()


def visualize_predictions(images, targets, predictions, max_images=3, dataset_type="Validation"):
    num_images = min(len(images), max_images)
    fig, axes = plt.subplots(2, num_images, figsize=(20, 10))

    for i in range(num_images):
        img_tensor = images[i]
        target = targets[i]
        prediction = predictions[i]

        img = T.ToPILImage()(img_tensor)

        # Истинные аннотации
        true_img = img.copy()
        draw_true = ImageDraw.Draw(true_img)
        for box in target["boxes"]:
            xmin, ymin, xmax, ymax = box.tolist()
            print(f"True Box: {xmin}, {ymin}, {xmax}, {ymax}")  # Отладочный вывод
            draw_true.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # Предсказания модели
        pred_img = img.copy()
        draw_pred = ImageDraw.Draw(pred_img)
        valid_preds = prediction["scores"] > 0.5
        pred_boxes = prediction["boxes"][valid_preds]
        for box in pred_boxes:
            xmin, ymin, xmax, ymax = box.tolist()
            print(f"Predicted Box: {xmin}, {ymin}, {xmax}, {ymax}")  # Отладочный вывод
            draw_pred.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)

        # Отображение изображений
        if num_images == 1:
            axes[0].imshow(true_img)
            axes[0].axis("off")
            axes[0].set_title(f"{dataset_type} - True Annotations {i}")
            axes[1].imshow(pred_img)
            axes[1].axis("off")
            axes[1].set_title(f"{dataset_type} - Predictions {i}")
        else:
            axes[0, i].imshow(true_img)
            axes[0, i].axis("off")
            axes[0, i].set_title(f"{dataset_type} - True Annotations {i}")
            axes[1, i].imshow(pred_img)
            axes[1, i].axis("off")
            axes[1, i].set_title(f"{dataset_type} - Predictions {i}")

    plt.tight_layout()
    plt.show()


def objective(trial):
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [4, 8]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["AdamW"]),
        "scheduler": trial.suggest_categorical("scheduler", ["CosineAnnealingLR"]),
        "model_version": trial.suggest_categorical("model_version", ["v2"]),
        "num_epochs": trial.suggest_int("num_epochs", 10, 20)
    }

    logging.info(f"Trial {trial.number}: {params}")

    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    num_classes = 2

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    transform = T.Compose([T.ToTensor()])

    train_dataset = CustomDataset(
        images_dir='../dataset_for_search_2/train/images',
        annotations_dir='../dataset_for_search_2/train/annotations',
        transforms=transform
    )

    val_dataset = CustomDataset(
        images_dir='../dataset_for_search_2/val/images',
        annotations_dir='../dataset_for_search_2/val/annotations',
        transforms=T.ToTensor()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"]
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=5)

    best_f1 = 0.0
    for epoch in range(params["num_epochs"]):
        model.train()
        total_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{params['num_epochs']}"):
            # Проверка и вывод информации о тензорах
            print("\nDebugging tensors:")
            print("Number of images:", len(images))
            print("Image shapes:")
            for i, img in enumerate(images):
                print(f"  Image {i}: {img.shape}")

            print("Targets keys and shapes:")
            for i, target in enumerate(targets):
                print(f"  Target {i}:")
                for key, value in target.items():
                    print(f"    {key}: {value.shape if isinstance(value, torch.Tensor) else value}")

            visualize_batch(images, targets, max_images=3)

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        logging.info(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")

        # Вычисление метрик на валидационном наборе
        metrics = calculate_metrics(model, val_loader, device)
        logging.info(f"Metrics: {metrics}")

        # Визуализация предсказаний на валидационном наборе
        model.eval()
        with torch.no_grad():
            val_images, val_targets = next(iter(val_loader))
            val_images = [img.to(device) for img in val_images]
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]
            val_predictions = model(val_images)

            val_images = [img.cpu() for img in val_images]
            val_targets = [{k: v.cpu() for k, v in t.items()} for t in val_targets]
            val_predictions = [{k: v.cpu() for k, v in p.items()} for p in val_predictions]

            visualize_predictions(val_images, val_targets, val_predictions, max_images=3)

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), f"best_model_trial_{trial.number}.pth")

        scheduler.step()

    return best_f1


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)
