import os
import torch
import torchvision
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


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'mask_rcnn_model_node.pth'
    test_image_dir = '../dataset_coco_neuro_3/images_neuro_3'
    test_annotations_dir = '../dataset_coco_neuro_3/ann'
    masks_folder = '../dataset_coco_neuro_3/masks'
    output_dir = '../predict_mask_rcnn_node_NEW'
    confidence_threshold = 0.5
    class_names = ['background', 'Node']
    colors = {
        'prediction': 'purple',
        'ground_truth': 'green'
    }


all_true_labels = []
all_pred_labels = []
all_scores = []
all_iou_scores = []


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


def calculate_mask_iou(pred_mask, true_mask):
    h, w = true_mask.shape
    scaled_pred_mask = zoom(pred_mask, (float(h) / pred_mask.shape[0], float(w) / pred_mask.shape[1]), order=0)
    scaled_pred_mask = (scaled_pred_mask > 0.5).astype(np.float32)

    intersection = np.logical_and(scaled_pred_mask, true_mask).sum()
    union = np.logical_or(scaled_pred_mask, true_mask).sum()
    return intersection / union if union != 0 else 0


def calculate_metrics(ground_truth, predictions, true_mask=None):
    if not ground_truth:
        return

    # Подготовка ground truth
    gt_boxes = [ann['bbox'] for ann in ground_truth]
    gt_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes]
    gt_labels = [ann['category_id'] for ann in ground_truth]

    # Подготовка предсказаний
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy() if 'masks' in predictions[0] else None

    # Фильтрация по порогу уверенности
    keep = pred_scores >= Config.confidence_threshold
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    pred_masks = pred_masks[keep] if pred_masks is not None else None

    # Для каждого GT объекта (Node)
    for gt_box, gt_label in zip(gt_boxes, gt_labels):
        all_true_labels.append(1)  # Все GT объекты - это Node (класс 1)

        best_iou = 0
        best_mask_iou = 0
        matched = False

        # Ищем совпадения с предсказаниями
        for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
            if pred_label == 1:
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    matched = True
                    matched_score = pred_score

        if matched:
            all_pred_labels.append(1)  # True Positive
            all_scores.append(matched_score)
            if best_iou > 0.5 and true_mask is not None and pred_masks is not None:
                pred_mask = pred_masks[np.argmax([calculate_iou(gt_box, pb) for pb in pred_boxes])][0]
                pred_mask = (pred_mask > 0.5).astype(np.float32)
                best_mask_iou = calculate_mask_iou(pred_mask, true_mask)
                all_iou_scores.append(best_mask_iou)
        else:
            all_pred_labels.append(0)  # False Negative
            all_scores.append(0.0)  # Добавляем нулевой скор для FN

    # Добавляем False Positives (предсказанные узлы, которым не соответствует GT)
    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
        if pred_label == 1:  # Если предсказан Node
            # Проверяем, не был ли этот бокс уже сопоставлен с GT
            is_matched = any(calculate_iou(gt_box, pred_box) > 0.5 for gt_box in gt_boxes)
            if not is_matched:
                all_true_labels.append(0)  # Это Background в действительности
                all_pred_labels.append(1)  # Но модель предсказала Node
                all_scores.append(pred_score)


def plot_confusion_matrix(save_dir):
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[1, 0])
    df_cm = pd.DataFrame(cm, index=['Node', 'Background'], columns=['Node', 'Background'])

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_normalized_confusion_matrix(save_dir):
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[1, 0], normalize='true')
    df_cm = pd.DataFrame(cm, index=['Node', 'Background'], columns=['Node', 'Background'])

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    save_path = os.path.join(save_dir, "normalized_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_precision_recall_curve(save_dir):
    precision, recall, thresholds = precision_recall_curve(all_true_labels, all_scores)
    ap = average_precision_score(all_true_labels, all_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    save_path = os.path.join(save_dir, "precision_recall_curve.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_iou_distribution(save_dir):
    plt.figure(figsize=(10, 5))
    plt.hist(all_iou_scores, bins=20, range=(0, 1))
    plt.title('Distribution of IoU Scores')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')

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
    if len(all_iou_scores) > 0:
        plot_iou_distribution(save_dir)


def save_image_with_predictions(image, predictions, class_names, output_path,
                                threshold=0.5, ground_truth=None, true_mask=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 528 / 100, 528 / 100))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1)

    plot_predictions(ax1, image, predictions, class_names, threshold, title="Predictions")
    plot_ground_truth(ax2, image, ground_truth, class_names, title="Ground Truth", true_mask=true_mask)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=100)
    plt.close()


def plot_predictions(ax, image, predictions, class_names, threshold, title):
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    labels = predictions[0]['labels'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()
    masks = predictions[0]['masks'].cpu().detach().numpy()

    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    masks = masks[keep]

    color_map = {
        'Node': 'purple',
        'background': 'yellow'
    }

    ax.imshow(image)
    ax.set_title(title)

    for box, label, mask in zip(boxes, labels, masks):
        class_name = class_names[label]
        color = color_map.get(class_name, 'black')

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
            if class_name == 'Node':
                image_with_mask[full_mask > 0.5] = [0.5, 0, 0.5]  # purple
            elif class_name == 'background':
                image_with_mask[full_mask > 0.5] = [1, 1, 0]  # yellow

            ax.imshow(image_with_mask, alpha=0.5)

    ax.axis('off')


def plot_ground_truth(ax, image, ground_truth, class_names, title, true_mask=None):
    ax.imshow(image)
    ax.set_title(title)

    color_map = {
        'Node': 'green',
        'background': 'blue'
    }

    for ann in ground_truth:
        box = ann['bbox']
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

        class_id = ann['category_id']
        class_name = class_names[class_id]
        color = color_map.get(class_name, 'black')

        ax.add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                          fill=False, edgecolor=color, linewidth=2))
        ax.text(box[0], box[1] - 10, class_name, color=color,
                fontsize=12, backgroundcolor='white')

    if true_mask is not None:
        image_with_mask = np.array(image).astype(np.float32) / 255.0
        image_with_mask[true_mask > 0] = [0, 0.5, 0]
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
        return None

    mask_path = os.path.join(masks_folder, matching_masks[0])

    try:
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)

        mask_array = (mask_array > 128).astype(np.float32)
        return mask_array

    except Exception as e:
        print(f"Ошибка загрузки маски: {e}")
        return None


def main():
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
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
                    image, predictions, Config.class_names, output_path,
                    threshold=Config.confidence_threshold,
                    ground_truth=ground_truth, true_mask=true_mask
                )

            except Exception as e:
                print(f"Ошибка при обработке {filename}: {str(e)}")

    if len(all_true_labels) > 0:
        metrics_dir = os.path.join(Config.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        plot_metrics(metrics_dir)

        mean_iou = np.mean([iou for iou in all_iou_scores if iou > 0])
        print(f"\nAverage Metrics:")
        print(f"Average Precision: {average_precision_score(all_true_labels, all_scores):.4f}")
        print(f"Mean IoU (for matched predictions): {mean_iou:.4f}")
        print(f"Mask IoU (for matched predictions): {np.mean(all_iou_scores):.4f}")
        print(f"Confusion Matrix saved in {metrics_dir}")
    else:
        print("No ground truth data found for metric calculation.")


if __name__ == "__main__":
    main()
