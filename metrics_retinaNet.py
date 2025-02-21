import os
import json
import numpy as np
import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import ToTensor
from PIL import Image
import cv2
from tqdm import tqdm


def calculate_iou_bbox(boxA, boxB):
    # определение координат пересечения
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # вычисление площади пересечения
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # вычисление площадей обоих прямоугольников
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def evaluate_retinanet_model(model, images_dir, annotations_dir, device, iou_threshold=0.5):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_name in tqdm(os.listdir(images_dir)):
        if not image_name.endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(images_dir, image_name)
        annotation_path = os.path.join(annotations_dir, f"{os.path.splitext(image_name)[0]}.json")

        if not os.path.exists(annotation_path):
            print(f"Annotation file not found for {image_name}")
            continue

        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)

        image = Image.open(image_path).convert("RGB")
        image_tensor = ToTensor()(image).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)[0]

        true_bboxes = []
        for ann in annotation_data.get("annotations", []):
            bbox = ann.get("bbox", None)
            if bbox:
                true_bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

        matched = set()
        pred_bboxes = predictions["boxes"].cpu().numpy()
        pred_scores = predictions["scores"].cpu().numpy()

        for pred_idx, pred_box in enumerate(pred_bboxes):
            if pred_scores[pred_idx] < 0.5:
                continue

            best_iou = 0
            best_true_idx = -1
            for true_idx, true_box in enumerate(true_bboxes):
                iou = calculate_iou_bbox(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_true_idx = true_idx

            if best_iou >= iou_threshold and best_true_idx not in matched:
                total_tp += 1
                matched.add(best_true_idx)
            else:
                total_fp += 1

        total_fn += len(true_bboxes) - len(matched)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def evaluate_retinanet_model_per_class(model, images_dir, annotations_dir, device, iou_threshold=0.5):
    model.eval()
    class_metrics = {}

    category_id_to_name = {}
    for annotation_file in os.listdir(annotations_dir):
        if annotation_file.endswith('.json'):
            annotation_path = os.path.join(annotations_dir, annotation_file)
            with open(annotation_path, 'r') as f:
                annotation_data = json.load(f)
            for category in annotation_data.get("categories", []):
                category_id_to_name[category["id"]] = category["name"]
            break

    for image_name in tqdm(os.listdir(images_dir)):
        if not image_name.endswith(('.jpg', '.png')):
            continue
        image_path = os.path.join(images_dir, image_name)
        annotation_path = os.path.join(annotations_dir, f"{os.path.splitext(image_name)[0]}.json")
        if not os.path.exists(annotation_path):
            print(f"Annotation file not found for {image_name}")
            continue

        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)

        image = Image.open(image_path).convert("RGB")
        image_tensor = ToTensor()(image).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)[0]

        true_bboxes = []
        true_labels = []
        for ann in annotation_data.get("annotations", []):
            bbox = ann.get("bbox", None)
            label = ann.get("category_id", None)
            if bbox and label is not None:
                true_bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                true_labels.append(label)

        pred_bboxes = predictions["boxes"].cpu().numpy()
        pred_scores = predictions["scores"].cpu().numpy()
        pred_labels = predictions["labels"].cpu().numpy()

        matched = set()

        for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_bboxes, pred_labels)):
            if pred_scores[pred_idx] < 0.5:
                continue

            best_iou = 0
            best_true_idx = -1

            for true_idx, (true_box, true_label) in enumerate(zip(true_bboxes, true_labels)):
                if true_label != pred_label:
                    continue
                iou = calculate_iou_bbox(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_true_idx = true_idx

            if best_iou >= iou_threshold and best_true_idx not in matched:
                # True Positive
                if pred_label not in class_metrics:
                    class_metrics[pred_label] = {"tp": 0, "fp": 0, "fn": 0}
                class_metrics[pred_label]["tp"] += 1
                matched.add(best_true_idx)
            else:
                # False Positive
                if pred_label not in class_metrics:
                    class_metrics[pred_label] = {"tp": 0, "fp": 0, "fn": 0}
                class_metrics[pred_label]["fp"] += 1

        # Подсчёт False Negatives для всех классов
        for true_label in true_labels:
            if true_label not in matched:
                if true_label not in class_metrics:
                    class_metrics[true_label] = {"tp": 0, "fp": 0, "fn": 0}
                class_metrics[true_label]["fn"] += 1

    # Вычисление метрик для каждого класса
    results = {}
    for cls_id, metrics in class_metrics.items():
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        )

        results[cls_id] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return results, category_id_to_name


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model_path = 'retinaNet/retinanet_model_node.pth'
    images_dir = 'dataset_coco_neuro_2/val/images'
    annotations_dir = 'dataset_coco_neuro_2/val/annotations'

    model = retinanet_resnet50_fpn(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    metrics_per_class, category_id_to_name = evaluate_retinanet_model_per_class(
        model, images_dir, annotations_dir, device
    )

    for cls_id, metrics in metrics_per_class.items():
        class_name = category_id_to_name.get(cls_id, f"Unknown Class ({cls_id})")
        print(f"Class: {class_name}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  True Positives: {metrics['tp']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")