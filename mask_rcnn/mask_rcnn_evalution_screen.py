import os
import torch
import torchvision
from sklearn.preprocessing import LabelBinarizer
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom
from collections import defaultdict


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'mask_rcnn_model_screen.pth'
    test_image_dir = '../dataset_coco_neuro_1/images_neuro_1'
    test_annotations_dir = '../dataset_coco_neuro_1/ann_neuro_1'
    masks_folder = '../dataset_coco_neuro_1/masks'
    output_dir = '../predict_mask_rcnn_screen_0.5'
    confidence_threshold = 0.5
    class_names = ['background', 'Thyroid tissue', 'Carotis']
    colors = {
        'Thyroid tissue': 'purple',
        'Carotis': 'green',
        'background': 'orange'
    }


all_true_labels = []
all_pred_labels = []
all_scores = []
all_iou_scores = defaultdict(list)
class_metrics = {
    'Thyroid tissue': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0},
    'Carotis': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0}
}


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def calculate_mask_iou(pred_mask, true_mask, class_id):
    h, w = true_mask.shape
    scaled_pred_mask = zoom(pred_mask, (float(h) / pred_mask.shape[0], float(w) / pred_mask.shape[1]), order=0)
    scaled_pred_mask = (scaled_pred_mask > 0.5).astype(np.float32)

    true_class_mask = (true_mask == class_id).astype(np.float32)

    intersection = np.logical_and(scaled_pred_mask, true_class_mask).sum()
    union = np.logical_or(scaled_pred_mask, true_class_mask).sum()
    return intersection / union if union != 0 else 0


def calculate_metrics(ground_truth, predictions, true_mask=None):
    if not ground_truth:
        return

    gt_boxes = [ann['bbox'] for ann in ground_truth]
    gt_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes]
    gt_labels = [ann['category_id'] for ann in ground_truth]

    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy() if 'masks' in predictions[0] else None

    keep = pred_scores >= Config.confidence_threshold
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    pred_masks = pred_masks[keep] if pred_masks is not None else None

    for gt_box, gt_label in zip(gt_boxes, gt_labels):
        class_name = Config.class_names[gt_label]
        all_true_labels.append(gt_label)

        best_iou = 0
        best_mask_iou = 0
        matched = False

        for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
            if pred_label == gt_label:
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    matched = True
                    matched_score = pred_score

        if matched:
            all_pred_labels.append(gt_label)
            all_scores.append(matched_score)
            class_metrics[class_name]['true_pos'] += 1

            if best_iou > 0.5 and true_mask is not None and pred_masks is not None:
                pred_mask = pred_masks[np.argmax([calculate_iou(gt_box, pb) for pb in pred_boxes])][0]
                pred_mask = (pred_mask > 0.5).astype(np.float32)
                best_mask_iou = calculate_mask_iou(pred_mask, true_mask, gt_label)
                all_iou_scores[class_name].append(best_mask_iou)
        else:
            all_pred_labels.append(0)
            all_scores.append(0.0)
            class_metrics[class_name]['false_neg'] += 1

    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
        is_matched = any(
            calculate_iou(gt_box, pred_box) > 0.5 and gt_label == pred_label
            for gt_box, gt_label in zip(gt_boxes, gt_labels)
        )

        if not is_matched and pred_label != 0:
            all_true_labels.append(0)
            all_pred_labels.append(pred_label)
            all_scores.append(pred_score)
            class_metrics[Config.class_names[pred_label]]['false_pos'] += 1


def plot_confusion_matrix(save_dir):
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1, 2])
    df_cm = pd.DataFrame(cm, index=Config.class_names, columns=Config.class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_normalized_confusion_matrix(save_dir):
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1, 2], normalize='true')
    df_cm = pd.DataFrame(cm, index=Config.class_names, columns=Config.class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    save_path = os.path.join(save_dir, "normalized_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_precision_recall_curve(save_dir):
    try:
        y_true = np.array(all_true_labels)
        y_scores = np.array(all_scores)

        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)

        plt.figure(figsize=(10, 5))

        for class_id, class_name in enumerate(Config.class_names):
            if class_id == 0:
                continue

            if class_id in y_true:
                class_true = (y_true == class_id).astype(int)
                precision, recall, _ = precision_recall_curve(class_true, y_scores)
                ap = average_precision_score(class_true, y_scores)

                plt.plot(recall, precision,
                         label=f'{class_name} (AP = {ap:.2f})',
                         color=Config.colors.get(class_name, 'black'))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(save_dir, "precision_recall_curves.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error plotting PR curves: {str(e)}")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        plt.title('Precision-Recall Curves (Failed)')

        save_path = os.path.join(save_dir, "pr_curves_error.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def plot_iou_distribution(save_dir):
    plt.figure(figsize=(10, 5))
    for class_name, iou_scores in all_iou_scores.items():
        if iou_scores:
            plt.hist(iou_scores, bins=20, range=(0, 1), alpha=0.5, label=class_name)

    plt.title('Distribution of IoU Scores by Class')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.legend()

    save_path = os.path.join(save_dir, "iou_distribution.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_metrics(save_dir):
    if len(all_true_labels) != len(all_scores):
        print(f"Warning: Mismatch in array lengths - true_labels: {len(all_true_labels)}, scores: {len(all_scores)}")
        return

    if len(all_true_labels) == 0:
        print("No data available for metrics")
        return

    plot_confusion_matrix(save_dir)
    plot_normalized_confusion_matrix(save_dir)
    plot_precision_recall_curve(save_dir)
    plot_iou_distribution(save_dir)

    # Вывод метрик по классам
    print("\nClass-specific Metrics:")
    for class_name, metrics in class_metrics.items():
        tp = metrics['true_pos']
        fp = metrics['false_pos']
        fn = metrics['false_neg']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        mean_iou = np.mean(all_iou_scores[class_name]) if all_iou_scores[class_name] else 0

        print(f"\n{class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Mean IoU: {mean_iou:.4f}")


def save_image_with_predictions(image, predictions, output_path, ground_truth=None, true_mask=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 528 / 100, 528 / 100))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1)

    plot_predictions(ax1, image, predictions, title="Predictions")
    plot_ground_truth(ax2, image, ground_truth, title="Ground Truth", true_mask=true_mask)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=100)
    plt.close()


def plot_predictions(ax, image, predictions, title):
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    labels = predictions[0]['labels'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()
    masks = predictions[0]['masks'].cpu().detach().numpy()

    keep = scores >= Config.confidence_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    masks = masks[keep]

    ax.imshow(image)
    ax.set_title(title)

    for box, label, mask in zip(boxes, labels, masks):
        class_name = Config.class_names[label]
        color = Config.colors.get(class_name, 'black')

        ax.add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                          fill=False, edgecolor=color, linewidth=2)
        )

        ax.text(box[0], box[1] - 10, class_name, color=color,
                fontsize=12, backgroundcolor='white')

        mask = mask[0]
        h, w = image.size[1], image.size[0]

        x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            box_mask = mask > 0.5
            box_mask_resized = zoom(
                box_mask,
                (float(y2 - y1) / box_mask.shape[0], float(x2 - x1) / box_mask.shape[1]),
                order=0
            )

            full_mask = np.zeros((h, w))
            full_mask[y1:y2, x1:x2] = box_mask_resized

            image_with_mask = np.array(image).astype(np.float32) / 255.0
            if class_name in Config.colors:
                color = Config.colors[class_name]
                if isinstance(color, str):
                    # Convert named color to RGB
                    from matplotlib.colors import to_rgb
                    color = to_rgb(color)
                image_with_mask[full_mask > 0.5] = color

            ax.imshow(image_with_mask, alpha=0.5)

    ax.axis('off')


def plot_ground_truth(ax, image, ground_truth, title, true_mask=None):
    ax.imshow(image)
    ax.set_title(title)

    if ground_truth is not None:
        for ann in ground_truth:
            box = ann['bbox']
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

            class_id = ann['category_id']
            class_name = Config.class_names[class_id]
            color = Config.colors.get(class_name, 'black')

            ax.add_patch(
                plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                              fill=False, edgecolor=color, linewidth=2))
            ax.text(box[0], box[1] - 10, class_name, color=color,
                    fontsize=12, backgroundcolor='white')

    if true_mask is not None:
        image_with_mask = np.array(image).astype(np.float32) / 255.0

        for class_id, class_name in enumerate(Config.class_names):
            if class_id == 0:
                continue

            class_mask = (true_mask == class_id)
            if class_mask.any() and class_name in Config.colors:
                color = Config.colors[class_name]
                if isinstance(color, str):
                    from matplotlib.colors import to_rgb
                    color = to_rgb(color)
                image_with_mask[class_mask] = color

        ax.imshow(image_with_mask, alpha=0.3)

    ax.axis('off')


def load_ground_truth(image_filename, annotations_folder):
    base_name = os.path.splitext(image_filename)[0]
    annotation_path = os.path.join(annotations_folder, f"{base_name}.json")

    if not os.path.exists(annotation_path):
        return None

    with open(annotation_path) as f:
        data = json.load(f)

    return data.get('annotations', [])


def load_true_mask(image_filename, masks_folder):
    base_name = os.path.splitext(image_filename)[0]
    base_name = base_name.split('_')[0]

    matching_masks = [f for f in os.listdir(masks_folder)
                      if f.startswith(f"{base_name}_") and f.endswith('.png')]

    if not matching_masks:
        print(f"Маски для {image_filename} не найдены.")
        return None

    full_mask = np.zeros((528, 528), dtype=np.uint8)

    for mask_file in matching_masks:
        mask_path = os.path.join(masks_folder, mask_file)

        try:
            mask = Image.open(mask_path).convert('L')
            mask_array = np.array(mask)
            mask_array = (mask_array > 128).astype(np.uint8)

            if 'Thyroid_tissue' in mask_file:
                class_value = 1
            elif 'Carotis' in mask_file:
                class_value = 2
            else:
                print(f"Неизвестный класс для маски: {mask_file}")
                continue

            full_mask[mask_array == 1] = class_value

        except Exception as e:
            print(f"Ошибка загрузки маски {mask_file}: {e}")
            continue

    return full_mask


def main():
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    num_classes = len(Config.class_names)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256,
                                                                                              num_classes)

    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.eval()

    model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
        min_size=528, max_size=528,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )

    transform = T.Compose([T.ToTensor()])
    os.makedirs(Config.output_dir, exist_ok=True)

    for filename in os.listdir(Config.test_image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(Config.test_image_dir, filename)

            try:
                image = Image.open(image_path).convert("RGB")
                ground_truth = load_ground_truth(filename, Config.test_annotations_dir)
                true_mask = load_true_mask(filename, Config.masks_folder)

                if not ground_truth:
                    continue

                image_tensor = transform(image).unsqueeze(0).to(Config.device)
                model.to(Config.device)

                with torch.no_grad():
                    predictions = model(image_tensor)

                calculate_metrics(ground_truth, predictions, true_mask)

                output_path = os.path.join(Config.output_dir, f"pred_{filename}")
                save_image_with_predictions(
                    image, predictions, output_path,
                    ground_truth=ground_truth, true_mask=true_mask
                )

            except Exception as e:
                print(f"Ошибка при обработке {filename}: {str(e)}")

    if len(all_true_labels) > 0:
        metrics_dir = os.path.join(Config.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        plot_metrics(metrics_dir)

        print("\nAverage Metrics:")

        try:
            y_true = np.array(all_true_labels)
            y_scores = np.array(all_scores)

            if len(y_scores.shape) == 1:
                y_scores = y_scores.reshape(-1, 1)

            if 1 in y_true:
                ap = average_precision_score(y_true, y_scores, pos_label=1)
                print(f"Average Precision: {ap:.4f}")
            else:
                print("Average Precision: N/A (no positive samples for Thyroid tissue)")
        except Exception as e:
            print(f"Error calculating Average Precision: {str(e)}")

        for class_name in Config.class_names[1:]:  # Пропускаем background
            if all_iou_scores[class_name]:
                print(f"Mean IoU for {class_name}: {np.mean(all_iou_scores[class_name]):.4f}")

        print(f"Confusion Matrix saved in {metrics_dir}")
    else:
        print("No ground truth data found for metric calculation.")


if __name__ == "__main__":
    main()
