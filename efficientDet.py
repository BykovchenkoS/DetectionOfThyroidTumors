import torch
import logging
import time
import csv
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import json
from effdet import get_efficientdet_config, create_model_from_config, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
import torchvision.transforms as T
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("efficientDet_screen.log"),
    logging.StreamHandler()
])


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

        if not os.path.exists(ann_path):
            raise ValueError(f"Annotation file not found for image: {img_filename}")

        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        category_map = {category['id']: category['name'] for category in annotation.get('categories', [])}
        annotations = annotation.get('annotations', [])

        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if self.transforms:
            img = self.transforms(img)

        target = {
            'bbox': torch.tensor(boxes, dtype=torch.float32),
            'cls': torch.tensor(labels, dtype=torch.long),
            'img_size': (img.shape[-2], img.shape[-1]),
            'img_filename': img_filename
        }

        return img, target, category_map


def custom_collate_fn(batch):
    images, targets, category_maps = zip(*batch)
    images = torch.stack([image for image in images])

    img_filenames = [target['img_filename'] for target in targets]

    max_objects = max(len(target['bbox']) for target in targets)

    processed_targets = []
    for target in targets:
        bbox = target['bbox']
        cls = target['cls']
        img_size = target['img_size']

        if len(bbox) < max_objects:
            padding = torch.zeros((max_objects - len(bbox), 4), dtype=torch.float32)
            bbox = torch.cat([bbox, padding], dim=0)
            cls = torch.cat([cls, torch.zeros((max_objects - len(cls),), dtype=torch.long)], dim=0)

        processed_target = {
            'bbox': bbox,
            'cls': cls,
            'img_size': torch.tensor(img_size, dtype=torch.float32),
            'img_scale': torch.tensor([1.0], dtype=torch.float32)
        }
        processed_targets.append(processed_target)

    targets = {
        'bbox': torch.stack([t['bbox'] for t in processed_targets]),
        'cls': torch.stack([t['cls'] for t in processed_targets]),
        'img_size': torch.stack([t['img_size'] for t in processed_targets]),
        'img_scale': torch.stack([t['img_scale'] for t in processed_targets]),
        'img_filenames': img_filenames
    }

    return images, targets, category_maps


def create_model(num_classes, model_name="tf_efficientdet_d0"):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (528, 528)
    net = create_model_from_config(config, pretrained=False)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )

    train_model = DetBenchTrain(net, config)
    predict_model = DetBenchPredict(net)

    return train_model, predict_model


def prepare_targets_for_model(processed_targets):
    targets = {
        'bbox': [],
        'cls': [],
        'img_size': [],
        'img_scale': []
    }

    for target in processed_targets:
        targets['bbox'].append(target['bbox'])
        targets['cls'].append(target['cls'])
        targets['img_size'].append(target['img_size'])
        targets['img_scale'].append(target['img_scale'])

    targets['bbox'] = torch.cat(targets['bbox'], dim=0)
    targets['cls'] = torch.cat(targets['cls'], dim=0)
    targets['img_size'] = torch.stack(targets['img_size'], dim=0)
    targets['img_scale'] = torch.stack(targets['img_scale'], dim=0)

    return targets


def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

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
        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    unique_recalls = np.unique(recalls)
    avg_precisions = []

    for r in unique_recalls:
        max_precision = np.max(precisions[recalls >= r])
        avg_precisions.append(max_precision)

    map_score = np.mean(avg_precisions)

    return map_score


def compute_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-6)


def evaluate_on_validation(model, val_loader, device, iou_threshold=0.5):
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets, _ in tqdm(val_loader, desc="Validation", unit='batch'):
            images = torch.stack([image.to(device) for image in images])

            detections = model(images)

            for i, detection in enumerate(detections):
                img_id = targets['img_filenames'][i]

                if isinstance(detection, dict):
                    boxes = detection.get('boxes', torch.empty((0, 4)))
                    scores = detection.get('scores', torch.empty(0))
                    labels = detection.get('labels', torch.empty(0))

                    num_boxes = boxes.size(0)

                    for j in range(num_boxes):
                        all_predictions.append({
                            "bbox": boxes[j].cpu().numpy(),
                            "score": scores[j].cpu().item(),
                            "category_id": labels[j].cpu().item(),
                            "image_id": img_id
                        })

                elif isinstance(detection, torch.Tensor):
                    boxes = detection[:, :4]
                    scores = detection[:, 4]
                    labels = detection[:, 5]

                    num_boxes = boxes.size(0)

                    for j in range(num_boxes):
                        all_predictions.append({
                            "bbox": boxes[j].cpu().numpy(),
                            "score": scores[j].cpu().item(),
                            "category_id": labels[j].cpu().item(),
                            "image_id": img_id
                        })

                else:
                    raise ValueError("Unexpected detection format")

                num_gt_boxes = targets['bbox'][i].size(0)

                for j in range(num_gt_boxes):
                    all_ground_truths.append({
                        "bbox": targets['bbox'][i][j].cpu().numpy(),
                        "category_id": targets['cls'][i][j].cpu().item(),
                        "image_id": img_id
                    })

    precision, recall = compute_precision_recall(all_predictions, all_ground_truths, iou_threshold)
    map_score = compute_map(all_predictions, all_ground_truths, iou_threshold)
    f1_score = compute_f1_score(precision, recall)

    return precision, recall, map_score, f1_score


transform = T.Compose([
    T.ToTensor()
])

dataset = CustomDataset(
    images_dir='dataset_coco_neuro_1/train/images',
    annotations_dir='dataset_coco_neuro_1/train/annotations',
    transforms=transform
)

data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)

val_dataset = CustomDataset(
    images_dir='dataset_coco_neuro_1/val/images',
    annotations_dir='dataset_coco_neuro_1/val/annotations',
    transforms=transform
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

img, target, category_map = dataset[0]
num_classes = len(category_map) + 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_model, predict_model = create_model(num_classes=num_classes, model_name="tf_efficientdet_d0")
train_model.to(device)
predict_model.to(device)

params = [p for p in train_model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=1e-4)

with open('efficientDet_detection_result_screen.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time', 'Train/Loss', 'LR/pg0', 'Precision', 'Recall', 'mAP', 'F1-Score'])
    num_epochs = 100

    for epoch in range(num_epochs):
        train_model.train()
        epoch_loss = 0
        start_time = time.time()

        for images, targets, _ in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            images = torch.stack([image.to(device) for image in images])
            processed_targets = {
                'bbox': targets['bbox'].to(device),
                'cls': targets['cls'].to(device),
                'img_size': targets['img_size'].to(device),
                'img_scale': targets['img_scale'].to(device)
            }
            optimizer.zero_grad()
            loss_dict = train_model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        precision, recall, map_score, f1_score = evaluate_on_validation(predict_model, val_loader, device)

        elapsed_time = time.time() - start_time
        logging.info(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, mAP={map_score:.4f}, F1-Score={f1_score:.4f}")
        writer.writerow([epoch + 1, elapsed_time, epoch_loss, optimizer.param_groups[0]['lr'], precision, recall, map_score, f1_score])

torch.save(train_model.state_dict(), 'efficientDet_detection_model_screen.pth')

