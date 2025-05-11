import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import cv2
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 25
PATIENCE = 3
MIN_DELTA = 1e-4
MASKS_ROOT = "dataset_sam_neuro_1/masks/"

CLASS_NAMES = ['Thyroid tissue', 'Carotis']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLASS_IDS = list(CLASS_TO_ID.values())


class Normalize:
    def __init__(self, mean=None, std=None):
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        return (tensor / 255.0 - self.mean.to(tensor.device)) / self.std.to(tensor.device)


normalize = Normalize()


class SAMDataset(Dataset):
    def __init__(self, images_dir, annotations_dir):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_name = img_name.replace('.jpg', '.json').replace('.png', '.json')
        ann_path = os.path.join(self.annotations_dir, ann_name)

        with open(ann_path, 'r') as f:
            ann = json.load(f)

        category_map = {cat['id']: cat['name'] for cat in ann['categories']}

        boxes = []
        masks = []
        classes = []

        for obj in ann["annotations"]:
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x + w, y + h])
            mask = np.zeros((1024, 1024), dtype=np.uint8)
            segmentation = obj["segmentation"]

            class_name = category_map[obj["category_id"]]
            class_id = CLASS_TO_ID[class_name]
            classes.append(class_id)

            if isinstance(segmentation, list):
                if len(segmentation) > 0 and isinstance(segmentation[0], list):
                    for seg in segmentation:
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 1)
                else:
                    poly = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)
            elif isinstance(segmentation, str):
                mask_path = os.path.join(MASKS_ROOT, os.path.basename(segmentation))
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Failed to load mask: {mask_path}")
                mask = (mask > 0).astype(np.uint8)
            else:
                raise ValueError(f"Неизвестный тип сегментации: {type(segmentation)}")

            masks.append(mask)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            classes = torch.tensor(classes, dtype=torch.long)

        masks = torch.tensor(np.stack(masks), dtype=torch.float32)
        image_tensor = torch.tensor(image).permute(2, 0, 1).float()
        image_tensor = normalize(image_tensor)

        return {
            "image": image_tensor,
            "boxes": boxes,
            "masks": masks,
            "classes": classes,
            "image_name": img_name
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    max_objects = max(len(item["boxes"]) for item in batch)

    padded_boxes = []
    padded_masks = []
    padded_classes = []

    for item in batch:
        boxes = item["boxes"]
        masks = item["masks"]
        classes = item["classes"]

        if len(boxes) < max_objects:
            pad_boxes = torch.zeros((max_objects - len(boxes), 4), dtype=torch.float32)
            boxes = torch.cat([boxes, pad_boxes])

            pad_masks = torch.zeros((max_objects - masks.shape[0], 1024, 1024), dtype=torch.float32)
            masks = torch.cat([masks, pad_masks])

            pad_classes = torch.zeros((max_objects - len(classes)), dtype=torch.long)
            classes = torch.cat([classes, pad_classes])

        padded_boxes.append(boxes)
        padded_masks.append(masks)
        padded_classes.append(classes)

    return {
        "image": images,
        "boxes": torch.stack(padded_boxes),
        "masks": torch.stack(padded_masks),
        "classes": torch.stack(padded_classes),
    }


def calculate_metrics(pred_masks, gt_masks, threshold=0.5):
    pred_masks = (pred_masks > threshold).float()
    tp = (pred_masks * gt_masks).sum()
    fp = (pred_masks * (1 - gt_masks)).sum()
    fn = ((1 - pred_masks) * gt_masks).sum()
    tn = ((1 - pred_masks) * (1 - gt_masks)).sum()

    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = (tp / (tp + fp + fn + 1e-8)).item()

    return accuracy.item(), precision.item(), recall.item(), iou


def calculate_map(pred_masks, gt_masks, thresholds=np.linspace(0.5, 0.95, 10)):
    pred_masks = pred_masks.detach().cpu().numpy()
    gt_masks = gt_masks.detach().cpu().numpy()

    aps = []
    for threshold in thresholds:
        pred_bin = (pred_masks > threshold).astype(np.float32)
        intersection = (pred_bin * gt_masks).sum(axis=(1, 2))
        union = (pred_bin + gt_masks).clip(0, 1).sum(axis=(1, 2))
        iou = intersection / (union + 1e-8)

        valid_iou = iou[iou > 0]
        if len(valid_iou) > 0:
            ap = valid_iou.mean()
        else:
            ap = 0.0
        aps.append(ap)

    map50 = aps[0]
    map95 = aps[-1]
    map_value = np.mean(aps)

    return map50, map95, map_value


def calculate_class_metrics(pred_masks, gt_masks, classes, class_ids, threshold=0.5):
    pred_masks = (pred_masks > threshold).float()
    class_metrics = {class_id: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for class_id in class_ids}

    for pred, gt, class_id in zip(pred_masks, gt_masks, classes):
        class_id = class_id.item()
        if class_id not in class_metrics:
            continue

        tp = (pred * gt).sum().item()
        fp = (pred * (1 - gt)).sum().item()
        fn = ((1 - pred) * gt).sum().item()
        tn = ((1 - pred) * (1 - gt)).sum().item()

        class_metrics[class_id]['tp'] += tp
        class_metrics[class_id]['fp'] += fp
        class_metrics[class_id]['fn'] += fn
        class_metrics[class_id]['tn'] += tn

    results = {}
    for class_id, metrics in class_metrics.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        tn = metrics['tn']

        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        results[class_id] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'iou': iou
        }

    return results


def save_epoch_metrics(epoch, train_metrics, val_metrics, filename="class_metrics_screen.csv"):
    file_exists = os.path.isfile(filename)
    fieldnames = ['epoch', 'phase', 'class_id', 'class_name', 'accuracy', 'precision', 'recall', 'iou', 'map50',
                  'map95', 'map']

    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for class_id, metrics in train_metrics.items():
            if metrics['count'] > 0:
                row = {
                    'epoch': epoch,
                    'phase': 'train',
                    'class_id': class_id,
                    'class_name': ID_TO_CLASS[class_id],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'iou': metrics['iou'],
                    'map50': metrics['map50'],
                    'map95': metrics['map95'],
                    'map': metrics['map']
                }
                writer.writerow(row)

        for class_id, metrics in val_metrics.items():
            if metrics['count'] > 0:
                row = {
                    'epoch': epoch,
                    'phase': 'val',
                    'class_id': class_id,
                    'class_name': ID_TO_CLASS[class_id],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'iou': metrics['iou'],
                    'map50': metrics['map50'],
                    'map95': metrics['map95'],
                    'map': metrics['map']
                }
                writer.writerow(row)


def evaluate_model(model, val_loader, device, class_ids):
    model.eval()
    total_loss = 0
    total_objects = 0

    class_metrics = {class_id: {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'iou': 0,
        'map50': 0,
        'map95': 0,
        'map': 0,
        'count': 0
    } for class_id in class_ids}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["image"].to(device)
            boxes = batch["boxes"].to(device)
            gt_masks = batch["masks"].to(device)
            classes = batch["classes"].to(device)
            image_embeddings = model.image_encoder(images)
            batch_loss = 0
            batch_objects = 0

            for i in range(images.shape[0]):
                img_boxes = boxes[i]
                valid_indices = (img_boxes.sum(dim=1) > 0)
                valid_boxes = img_boxes[valid_indices]
                valid_classes = classes[i][valid_indices]

                if len(valid_boxes) == 0:
                    continue

                sparse_emb, dense_emb = model.prompt_encoder(
                    points=None,
                    boxes=valid_boxes,
                    masks=None,
                )

                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )

                pred_masks = F.interpolate(
                    low_res_masks,
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                valid_gt_masks = gt_masks[i][valid_indices]
                loss = criterion(pred_masks, valid_gt_masks)
                batch_loss += loss.item() * len(valid_boxes)
                batch_objects += len(valid_boxes)

                metrics = calculate_class_metrics(pred_masks, valid_gt_masks, valid_classes, class_ids)

                for class_id, m in metrics.items():
                    class_metrics[class_id]['accuracy'] += m['accuracy']
                    class_metrics[class_id]['precision'] += m['precision']
                    class_metrics[class_id]['recall'] += m['recall']
                    class_metrics[class_id]['iou'] += m['iou']
                    class_metrics[class_id]['count'] += 1

                    pred = pred_masks[valid_classes == class_id]
                    gt = valid_gt_masks[valid_classes == class_id]
                    if len(pred) > 0 and len(gt) > 0:
                        map50, map95, map_value = calculate_map(pred, gt)
                        class_metrics[class_id]['map50'] += map50
                        class_metrics[class_id]['map95'] += map95
                        class_metrics[class_id]['map'] += map_value

            if batch_objects > 0:
                total_loss += batch_loss
                total_objects += batch_objects

    if total_objects > 0:
        total_loss /= total_objects

        for class_id in class_metrics:
            if class_metrics[class_id]['count'] > 0:
                for metric in ['accuracy', 'precision', 'recall', 'iou', 'map50', 'map95', 'map']:
                    class_metrics[class_id][metric] /= class_metrics[class_id]['count']

    return total_loss, class_metrics


sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(DEVICE)

# Замораживаем все, кроме mask_decoder
for name, param in sam.named_parameters():
    if "mask_decoder" not in name:
        param.requires_grad = False

train_dataset = SAMDataset(
    images_dir="dataset_sam_neuro_1/train/images",
    annotations_dir="dataset_sam_neuro_1/train/annotations"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataset = SAMDataset(
    images_dir="dataset_sam_neuro_1/val/images",
    annotations_dir="dataset_sam_neuro_1/val/annotations"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

optimizer = Adam(sam.mask_decoder.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

best_val_loss = float('inf')
no_improve = 0

for epoch in range(EPOCHS):
    sam.train()
    epoch_loss = 0
    total_objects = 0
    train_class_metrics = {class_id: {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'iou': 0,
        'map50': 0,
        'map95': 0,
        'map': 0,
        'count': 0
    } for class_id in CLASS_IDS}

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
        images = batch["image"].to(DEVICE)
        boxes = batch["boxes"].to(DEVICE)
        gt_masks = batch["masks"].to(DEVICE)
        classes = batch["classes"].to(DEVICE)
        optimizer.zero_grad()

        with torch.no_grad():
            image_embeddings = sam.image_encoder(images)

        total_loss = 0
        batch_objects = 0

        for i in range(images.shape[0]):
            img_boxes = boxes[i]
            valid_indices = (img_boxes.sum(dim=1) > 0)
            valid_boxes = img_boxes[valid_indices]
            valid_classes = classes[i][valid_indices]

            if len(valid_boxes) == 0:
                continue

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=valid_boxes,
                masks=None,
            )

            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

            pred_masks = F.interpolate(
                low_res_masks,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            valid_gt_masks = gt_masks[i][valid_indices]
            loss = criterion(pred_masks, valid_gt_masks)
            total_loss += loss
            batch_objects += len(valid_boxes)

            metrics = calculate_class_metrics(pred_masks, valid_gt_masks, valid_classes, CLASS_IDS)

            for class_id, m in metrics.items():
                train_class_metrics[class_id]['accuracy'] += m['accuracy']
                train_class_metrics[class_id]['precision'] += m['precision']
                train_class_metrics[class_id]['recall'] += m['recall']
                train_class_metrics[class_id]['iou'] += m['iou']
                train_class_metrics[class_id]['count'] += 1

                pred = pred_masks[valid_classes == class_id]
                gt = valid_gt_masks[valid_classes == class_id]
                if len(pred) > 0 and len(gt) > 0:
                    map50, map95, map_value = calculate_map(pred, gt)
                    train_class_metrics[class_id]['map50'] += map50
                    train_class_metrics[class_id]['map95'] += map95
                    train_class_metrics[class_id]['map'] += map_value

        if batch_objects > 0:
            loss = total_loss / batch_objects
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_objects
            total_objects += batch_objects

    if total_objects > 0:
        epoch_loss /= total_objects
        for class_id in train_class_metrics:
            if train_class_metrics[class_id]['count'] > 0:
                for metric in ['accuracy', 'precision', 'recall', 'iou', 'map50', 'map95', 'map']:
                    train_class_metrics[class_id][metric] /= train_class_metrics[class_id]['count']

    val_loss, val_class_metrics = evaluate_model(sam, val_loader, DEVICE, CLASS_IDS)
    scheduler.step(val_loss)  # Обновляем learning rate на основе val_loss

    # Проверка на улучшение валидационного лосса
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        no_improve = 0
        torch.save({
            'mask_decoder_state_dict': sam.mask_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'val_loss': val_loss,
        }, "sam_best_screen.pth")
        print("\nЛучшая модель сохранена!")
    else:
        no_improve += 1
        print(f"\nНет улучшения {no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            print("Ранняя остановка!")
            break

    save_epoch_metrics(epoch + 1, train_class_metrics, val_class_metrics)

    print(f"\nEpoch {epoch + 1}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    for class_id in CLASS_IDS:
        train_metrics = train_class_metrics[class_id]
        val_metrics = val_class_metrics[class_id]

        if train_metrics['count'] > 0:
            print(f"Class {ID_TO_CLASS[class_id]} - Train: "
                  f"Acc: {train_metrics['accuracy']:.4f}, Prec: {train_metrics['precision']:.4f}, "
                  f"Rec: {train_metrics['recall']:.4f}, IoU: {train_metrics['iou']:.4f}, "
                  f"mAP: {train_metrics['map']:.4f}")

        if val_metrics['count'] > 0:
            print(f"Class {ID_TO_CLASS[class_id]} - Val: "
                  f"Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, "
                  f"Rec: {val_metrics['recall']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                  f"mAP: {val_metrics['map']:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'mask_decoder_state_dict': sam.mask_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'val_loss': val_loss,
        }, "sam_best_screen.pth")
        print("\nЛучшая модель сохранена!")

torch.save({
    'mask_decoder_state_dict': sam.mask_decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch + 1,
}, "sam_finetuned_screen.pth")

print("\nОбучение завершено! Модель сохранена в sam_final_screen.pth")
