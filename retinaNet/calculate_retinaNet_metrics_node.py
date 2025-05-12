import csv
import os
from pathlib import Path
import cv2
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'retinaNet/retinanet_model_node.pth'
    test_image_dir = 'dataset_coco_neuro_3/val/images'
    test_annotations_dir = 'dataset_coco_neuro_3/val/annotations'
    output_dir = 'predict_retinanet_node_metrics04'
    confidence_threshold = 0.4
    class_names = {
        1: 'Node',
        2: 'Background'
    }
    colors = {
        'prediction': 'red',
        'ground_truth': 'green'
    }


def load_model():
    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
    in_channels = model.head.classification_head.conv[0].out_channels

    model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=model.head.classification_head.num_anchors,
        num_classes=len(Config.class_names),
        norm_layer=torch.nn.BatchNorm2d
    )

    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.eval()
    return model.to(Config.device)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(Config.device)
    return image, image_tensor.unsqueeze(0)


def load_annotations(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(Config.test_annotations_dir, f"{base_name}.json")

    if not os.path.exists(annotation_path):
        return []

    with open(annotation_path) as f:
        data = json.load(f)

    return data.get('annotations', [])


def predict_objects(model, image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    filtered_predictions = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= Config.confidence_threshold:
            box = box.cpu().numpy().astype(int)
            label = label.item()
            score = score.item()
            filtered_predictions.append((label, *box, score))

    return filtered_predictions


def calculate_iou(box1, box2):
    x1_pred, y1_pred, x2_pred, y2_pred = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    x_left = max(x1_pred, x1_gt)
    y_top = max(y1_pred, y1_gt)
    x_right = min(x2_pred, x2_gt)
    y_bottom = min(y2_pred, y2_gt)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    iou = intersection_area / float(pred_area + gt_area - intersection_area)
    return iou


def parse_annotations(annotations, image_width, image_height):
    ground_truth = []
    for ann in annotations:
        cls = ann['category_id']
        x_min, y_min, width, height = ann['bbox']
        x_max = x_min + width
        y_max = y_min + height
        ground_truth.append((cls, int(x_min), int(y_min), int(x_max), int(y_max)))
    return ground_truth


def match_predictions_with_ground_truth(predictions, ground_truth):
    matched_results = []
    for pred in predictions:
        pred_cls, x1_pred, y1_pred, x2_pred, y2_pred, conf = pred
        best_iou = 0.0
        best_gt = None

        for gt in ground_truth:
            gt_cls, x1_gt, y1_gt, x2_gt, y2_gt = gt
            iou = calculate_iou((x1_pred, y1_pred, x2_pred, y2_pred), (x1_gt, y1_gt, x2_gt, y2_gt))

            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_gt:
            gt_cls, _, _, _, _ = best_gt
            matched_results.append((pred_cls, gt_cls, best_iou))

    return matched_results


def plot_boxes(ax, boxes, color, title, show_confidence=False):
    for box in boxes:
        if show_confidence:
            cls, x1, y1, x2, y2, conf = box
            label = f"{Config.class_names[cls]} {conf:.2f}"
        else:
            cls, x1, y1, x2, y2 = box
            label = Config.class_names[cls]

        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        ax.text(x1, y1 - 10, label, fontsize=12, color='white', bbox=dict(facecolor=color, alpha=0.7))

    ax.set_title(title)
    ax.axis('off')


def visualize_predictions(image, predictions, ground_truth=None):
    fig, axes = plt.subplots(1, 2 if ground_truth else 1, figsize=(20, 10))
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    if ground_truth:
        ax_pred = axes[0]
        ax_pred.imshow(image_rgb)
        plot_boxes(ax_pred, predictions, 'green', 'Predictions', show_confidence=True)

        ax_gt = axes[1]
        ax_gt.imshow(image_rgb)
        plot_boxes(ax_gt, ground_truth, 'red', 'Ground Truth')

    else:
        ax = axes
        ax.imshow(image_rgb)
        plot_boxes(ax, predictions, 'green', 'Predictions', show_confidence=True)

    plt.tight_layout()
    return fig


def calculate_map(predictions_list, ground_truth_list):
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        class_metrics=False  # Отключаем отдельные метрики для классов
    )

    preds = []
    targets = []

    for predictions, ground_truth in zip(predictions_list, ground_truth_list):
        pred_boxes = [pred[1:5] for pred in predictions]
        pred_scores = [pred[5] for pred in predictions]
        pred_labels = [pred[0] for pred in predictions]

        gt_boxes = [gt[1:5] for gt in ground_truth]
        gt_labels = [gt[0] for gt in ground_truth]

        preds.append({
            'boxes': torch.tensor(pred_boxes),
            'scores': torch.tensor(pred_scores),
            'labels': torch.tensor(pred_labels)
        })

        targets.append({
            'boxes': torch.tensor(gt_boxes),
            'labels': torch.tensor(gt_labels)
        })

    metric.update(preds, targets)
    result = metric.compute()

    return result


def save_metrics_to_csv(metrics, output_path):
    with open(output_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Metric Type', 'Class', 'mAP50', 'mAP50-95'])

        # Все классы
        writer.writerow([
            'All classes',
            'Node',
            f"{metrics['map_50']:.8f}",
            f"{metrics['map']:.8f}"
        ])


os.makedirs(Config.output_dir, exist_ok=True)
model = load_model()

all_predictions = []
all_ground_truths = []

with open(os.path.join(Config.output_dir, 'iou_results.csv'), mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['image_name', 'predicted_class', 'ground_truth_class', 'iou'])

    for image_path in tqdm(Path(Config.test_image_dir).glob('*.jpg'), desc="Processing images"):
        image, image_tensor = load_image(image_path)
        predictions = predict_objects(model, image_tensor)
        annotations = load_annotations(image_path)
        ground_truth = parse_annotations(annotations, image.width, image.height)

        matched_results = match_predictions_with_ground_truth(predictions, ground_truth)

        for pred_cls, gt_cls, iou in matched_results:
            writer.writerow([image_path.name, Config.class_names[pred_cls], Config.class_names[gt_cls], iou])

        all_predictions.append(predictions)
        all_ground_truths.append(ground_truth)
        fig = visualize_predictions(image, predictions, ground_truth)
        output_path = os.path.join(Config.output_dir, f"{image_path.stem}_result.png")
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)

print("Processing completed.")

map_results = calculate_map(all_predictions, all_ground_truths)

metrics = {
    'map_50': map_results['map_50'].item(),
    'map': map_results['map'].item()
}

metrics_csv_path = os.path.join(Config.output_dir, 'metrics.csv')
save_metrics_to_csv(metrics, metrics_csv_path)

print(f"Metrics saved to {metrics_csv_path}")
