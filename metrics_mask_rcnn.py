import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn  import MaskRCNNPredictor
from torchvision.transforms import ToTensor
from PIL import Image
import cv2
from tqdm import tqdm
from coco_for_pytorch import category_map
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc


def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def load_external_mask(mask_path, image_size, masks_dir):
    full_mask_path = os.path.join(masks_dir, os.path.basename(mask_path))

    if not os.path.exists(full_mask_path):
        raise FileNotFoundError(f"Mask file not found: {full_mask_path}")

    mask = Image.open(full_mask_path).convert("L")
    mask = mask.resize(image_size, Image.NEAREST)
    mask = np.array(mask) > 0
    return mask.astype(np.uint8)


def polygon_to_mask(polygon, image_size):
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    points = np.array(polygon, dtype=np.int32).reshape((-1, 2))
    cv2.fillPoly(mask, [points], 1)

    return mask


def plot_confusion_matrix(metrics_by_class, classes, normalize=False, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes))

    for cls_id, metrics in metrics_by_class.items():
        if cls_id not in classes:
            continue

        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        class_idx = classes.index(cls_id)

        cm[class_idx, class_idx] = tp  # True Positives
        for other_cls in classes:
            if other_cls != cls_id:
                other_idx = classes.index(other_cls)
                cm[other_idx, class_idx] += fn  # False Negatives
                cm[class_idx, other_idx] += fp  # False Positives

    if normalize:
        cm = cm.astype('float')
        row_sums = cm.sum(axis=1)
        cm[row_sums != 0] /= row_sums[row_sums != 0][:, np.newaxis]
        fmt = '.2f'
    else:
        cm = cm.astype('int')
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    filename = os.path.join(save_dir, f"confusion_matrix{'_normalized' if normalize else ''}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_f1_curve(precision, recall, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    precision = np.array(precision)
    recall = np.array(recall)

    # Вычисляем F1 Score
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores[np.isnan(f1_scores)] = 0

    plt.figure(figsize=(8, 6))
    plt.plot(recall, f1_scores, marker='o', label="F1 Score")
    plt.title("F1 Curve")
    plt.xlabel("Recall")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    filename = os.path.join(save_dir, "f1_curve.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_curve(precision, recall, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    precision = np.array(precision)
    recall = np.array(recall)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o', label="Precision")
    plt.title("Precision Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)

    filename = os.path.join(save_dir, "precision_curve.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_recall_curve(recall, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, marker='o', label="Recall")
    plt.title("Recall Curve")
    plt.xlabel("Thresholds")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)

    filename = os.path.join(save_dir, "recall_curve.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve(precision, recall, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    precision = np.array(precision)
    recall = np.array(recall)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o', label="PR Curve")
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)

    filename = os.path.join(save_dir, "pr_curve.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model, images_dir, annotations_dir, masks_dir, device, iou_threshold=0.5):
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
        image_size = (image.width, image.height)

        with torch.no_grad():
            predictions = model(image_tensor)[0]

        true_masks = []
        for ann in annotation_data.get("annotations", []):
            segmentation = ann.get("segmentation", None)
            if isinstance(segmentation, list):
                mask = polygon_to_mask(segmentation, image_size)
            elif isinstance(segmentation, str):
                mask = load_external_mask(segmentation, image_size, masks_dir)
            else:
                print(f"Unsupported segmentation format for {image_name}")
                continue

            true_masks.append(mask)

        matched = set()
        for pred_idx, pred_mask in enumerate(predictions["masks"]):
            pred_mask = pred_mask.cpu().numpy()[0] > 0.5
            best_iou = 0
            best_true_idx = -1
            for true_idx, true_mask in enumerate(true_masks):
                iou = calculate_iou(pred_mask, true_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_true_idx = true_idx

            if best_iou >= iou_threshold and best_true_idx not in matched:
                total_tp += 1
                matched.add(best_true_idx)
            else:
                total_fp += 1

        total_fn += len(true_masks) - len(matched)

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


def evaluate_model_by_class(model, images_dir, annotations_dir, masks_dir, device, iou_threshold=0.5):
    model.eval()
    class_metrics = {cls_id: {"tp": 0, "fp": 0, "fn": 0} for cls_id in category_map.keys()}

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
        image_size = (image.width, image.height)

        with torch.no_grad():
            predictions = model(image_tensor)[0]

        true_masks_by_class = {}
        for ann in annotation_data.get("annotations", []):
            cls_id = ann.get("category_id", None)
            segmentation = ann.get("segmentation", None)

            if cls_id is None or segmentation is None or cls_id not in category_map:
                continue
            if isinstance(segmentation, list):
                mask = polygon_to_mask(segmentation, image_size)
            elif isinstance(segmentation, str):
                mask = load_external_mask(segmentation, image_size, masks_dir)
            else:
                print(f"Unsupported segmentation format for {image_name}")
                continue

            if cls_id not in true_masks_by_class:
                true_masks_by_class[cls_id] = []
            true_masks_by_class[cls_id].append(mask)

        for pred_idx, (pred_mask, pred_label) in enumerate(zip(predictions["masks"], predictions["labels"])):
            pred_mask = pred_mask.cpu().numpy()[0] > 0.5
            cls_id = pred_label.item()
            if cls_id not in class_metrics:
                continue
            best_iou = 0
            best_true_idx = -1
            if cls_id in true_masks_by_class:
                for true_idx, true_mask in enumerate(true_masks_by_class[cls_id]):
                    iou = calculate_iou(pred_mask, true_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_true_idx = true_idx
            if best_iou >= iou_threshold and best_true_idx != -1:
                class_metrics[cls_id]["tp"] += 1
                true_masks_by_class[cls_id].pop(best_true_idx)
            else:
                class_metrics[cls_id]["fp"] += 1

        for cls_id, masks in true_masks_by_class.items():
            if cls_id not in class_metrics:
                continue
            class_metrics[cls_id]["fn"] += len(masks)

    results_by_class = {}
    for cls_id, stats in class_metrics.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results_by_class[cls_id] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    return results_by_class


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = len(category_map) + 1

    model_path = 'mask_rcnn_model_screen.pth'
    images_dir = 'dataset_coco_neuro_1/val/images'
    annotations_dir = 'dataset_coco_neuro_1/val/annotations'
    masks_dir = 'dataset_coco_neuro_1/masks'

    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    metrics_by_class = evaluate_model_by_class(model, images_dir, annotations_dir, masks_dir, device)

    for cls_id, metrics in metrics_by_class.items():
        class_name = category_map.get(cls_id, f"Class {cls_id}")
        print(f"\nClass: {class_name}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"True Positives: {metrics['tp']}")
        print(f"False Positives: {metrics['fp']}")
        print(f"False Negatives: {metrics['fn']}")

    save_dir = "mask_rcnn/plots"

    precisions = []
    recalls = []
    f1_scores = []
    class_names = []

    for cls_id, metrics in metrics_by_class.items():
        class_names.append(category_map.get(cls_id, f"Class {cls_id}"))
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1_scores.append(metrics["f1_score"])

    plot_confusion_matrix(metrics_by_class, class_names, normalize=False, save_dir=save_dir)

    plot_pr_curve(precisions, recalls, save_dir=save_dir)
    plot_f1_curve(precisions, recalls, save_dir=save_dir)
    plot_precision_curve(precisions, recalls, save_dir=save_dir)
    plot_recall_curve(recalls, save_dir=save_dir)