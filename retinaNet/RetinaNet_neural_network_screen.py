import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.ops import box_iou
import torchvision.transforms as T
from tqdm import tqdm
from coco_for_pytorch import CustomDataset
import csv
import time
import logging
import numpy as np
from collections import defaultdict
from torchvision.ops import nms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler("training_retinanet.log"),
                              logging.StreamHandler()])


def main():
    def create_model(num_classes):
        model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
        in_channels = model.head.classification_head.conv[0].out_channels

        model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
            norm_layer=torch.nn.BatchNorm2d
        )
        return model

    num_classes = 3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = create_model(num_classes)
    model.to(device)

    train_transform = T.Compose([
        T.ToTensor(),

        T.Lambda(lambda x: x.expand(3, -1, -1)),

        T.RandomHorizontalFlip(p=0.3),
        T.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0,
            hue=0
        ),

        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.expand(3, -1, -1)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CustomDataset(
        images_dir='../dataset_coco_neuro_1/train/images',
        annotations_dir='../dataset_coco_neuro_1/train/annotations',
        transforms=train_transform
    )

    val_dataset = CustomDataset(
        images_dir='../dataset_coco_neuro_1/val/images',
        annotations_dir='../dataset_coco_neuro_1/val/annotations',
        transforms=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0,
        pin_memory=True
    )

    optimizer = optim.SGD(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    def calculate_metrics(predictions, targets, iou_threshold=0.5):
        metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0}

        for pred, target in zip(predictions, targets):
            keep = nms(pred['boxes'], pred['scores'], iou_threshold=0.4)
            pred_boxes = pred['boxes'][keep]
            pred_scores = pred['scores'][keep]

            confidence_mask = pred_scores > 0.3
            pred_boxes = pred_boxes[confidence_mask]

            gt_boxes = target['boxes']
            num_gt = len(gt_boxes)

            if num_gt == 0:
                metrics['fp'] += len(pred_boxes)
                continue

            if len(pred_boxes) == 0:
                metrics['fn'] += num_gt
                continue

            iou_matrix = box_iou(pred_boxes, gt_boxes)

            gt_matched = set()
            pred_matched = set()

            for gt_idx in range(num_gt):
                best_pred_idx = iou_matrix[:, gt_idx].argmax()
                best_iou = iou_matrix[best_pred_idx, gt_idx]

                if best_iou >= iou_threshold and best_pred_idx not in pred_matched:
                    metrics['tp'] += 1
                    gt_matched.add(gt_idx)
                    pred_matched.add(best_pred_idx)
                    metrics['iou_sum'] += best_iou.item()

            metrics['fp'] += len(pred_boxes) - len(pred_matched)

            metrics['fn'] += num_gt - len(gt_matched)

        return metrics

    def evaluate(model, data_loader, device):
        model.eval()
        metrics = defaultdict(float)

        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions = model(images)
                batch_metrics = calculate_metrics(predictions, targets)

                for key in batch_metrics:
                    metrics[key] += batch_metrics[key]

        precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-8)
        recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-8)
        avg_iou = metrics['iou_sum'] / (metrics['tp'] + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': avg_iou,
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn']
        }

    try:
        with open('retinanet_metrics.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'epoch', 'time', 'train_loss', 'train_box_loss', 'train_cls_loss',
                'val_precision', 'val_recall', 'val_f1', 'val_iou', 'lr',
                'tp', 'fp', 'fn'
            ])

            best_f1 = 0.0
            num_epochs = 50

            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                epoch_box_loss = 0.0
                epoch_cls_loss = 0.0
                start_time = time.time()

                for images, targets in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}"):
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    optimizer.zero_grad()
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    losses.backward()
                    optimizer.step()

                    epoch_loss += losses.item()
                    epoch_box_loss += loss_dict['bbox_regression'].item()
                    epoch_cls_loss += loss_dict['classification'].item()

                epoch_time = time.time() - start_time

                val_metrics = evaluate(model, val_loader, device)
                scheduler.step(val_metrics['f1'])

                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    torch.save(model.state_dict(), 'best_retinanet_model.pth')

                avg_loss = epoch_loss / len(train_loader)
                avg_box_loss = epoch_box_loss / len(train_loader)
                avg_cls_loss = epoch_cls_loss / len(train_loader)

                writer.writerow([
                    epoch + 1, epoch_time,
                    avg_loss, avg_box_loss, avg_cls_loss,
                    val_metrics['precision'], val_metrics['recall'],
                    val_metrics['f1'], val_metrics['avg_iou'],
                    optimizer.param_groups[0]['lr'],
                    val_metrics['tp'], val_metrics['fp'], val_metrics['fn']
                ])
                file.flush()

                logging.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {avg_loss:.4f} | "
                    f"Val F1: {val_metrics['f1']:.3f} | "
                    f"Time: {epoch_time:.1f}s"
                )

        torch.save(model.state_dict(), 'retinanet_model_screen_new.pth')
        logging.info(f"Training complete. Best F1: {best_f1:.4f}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
