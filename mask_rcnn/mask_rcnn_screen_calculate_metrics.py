import csv
import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom
from collections import defaultdict
from torchmetrics.detection import MeanAveragePrecision


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'mask_rcnn/mask_rcnn_model_screen.pth'
    test_image_dir = 'dataset_coco_neuro_1/val/images'
    test_annotations_dir = 'dataset_coco_neuro_1/val/annotations'
    masks_folder = 'dataset_coco_neuro_1/masks'
    output_dir = 'predict_mask_rcnn_screen_metrics'

    confidence_threshold = 0.5

    class_names = ['background', 'Thyroid tissue', 'Carotis']
    colors = {
        'Thyroid tissue': 'purple',
        'Carotis': 'green',
        'background': 'orange'
    }


map_metric = MeanAveragePrecision(
    box_format='xyxy',
    iou_type='bbox',
    iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    rec_thresholds=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    max_detection_thresholds=[1, 10, 100],
    class_metrics=True
)

all_true_labels = []
all_pred_labels = []
all_scores = []
all_iou_scores = defaultdict(list)
class_metrics = {
    'Thyroid tissue': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0},
    'Carotis': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0}
}


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


def calculate_iou(pred_mask, true_mask, class_id):
    pred_class = (pred_mask == class_id).astype(int)
    true_class = (true_mask == class_id).astype(int)

    intersection = np.logical_and(pred_class, true_class).sum()
    union = np.logical_or(pred_class, true_class).sum()

    return intersection / union if union > 0 else 0.0


def convert_to_coco_format(predictions, image_id, height, width):
    pred_boxes = []
    pred_scores = []
    pred_labels = []

    boxes = predictions[0]['boxes'].cpu().detach()
    labels = predictions[0]['labels'].cpu().detach()
    scores = predictions[0]['scores'].cpu().detach()

    keep = scores >= Config.confidence_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels
    }


def convert_ground_truth_to_coco(ground_truth, image_id, height, width):
    gt_boxes = []
    gt_labels = []

    for ann in ground_truth:
        box = ann['bbox']
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        gt_boxes.append(box)
        gt_labels.append(ann['category_id'])

    return {
        'boxes': torch.tensor(gt_boxes, dtype=torch.float32),
        'labels': torch.tensor(gt_labels, dtype=torch.int64)
    }


def calculate_map(all_preds, all_targets):
    map_50_metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        iou_thresholds=[0.5],
        class_metrics=True
    )

    map_50_95_metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        iou_thresholds=np.linspace(0.5, 0.95, 10).tolist(),
        class_metrics=True
    )

    map_50_metric.update(all_preds, all_targets)
    map_50_95_metric.update(all_preds, all_targets)

    results_50 = map_50_metric.compute()
    results_50_95 = map_50_95_metric.compute()

    map50 = results_50['map'].item()
    map50_95 = results_50_95['map'].item()

    class_wise_map50 = [x.item() for x in results_50.get('map_per_class', [])] if 'map_per_class' in results_50 else []
    class_wise_map50_95 = [x.item() for x in
                           results_50_95.get('map_per_class', [])] if 'map_per_class' in results_50_95 else []

    map_50_metric.reset()
    map_50_95_metric.reset()

    return map50, map50_95, class_wise_map50, class_wise_map50_95


def process_image_predictions(predictions, true_mask, image_name):
    if true_mask is None:
        return

    pred_mask = np.zeros_like(true_mask)
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    labels = predictions[0]['labels'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()
    masks = predictions[0]['masks'].cpu().detach().numpy()
    keep = scores >= Config.confidence_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    masks = masks[keep]
    iou_per_object = []

    for idx, (box, label, mask) in enumerate(zip(boxes, labels, masks)):
        mask = mask[0] > 0.5
        h, w = true_mask.shape

        x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        box_width = x2 - x1
        box_height = y2 - y1

        resized_mask = zoom(mask, (box_height / mask.shape[0], box_width / mask.shape[1]), order=0)
        resized_mask = resized_mask > 0.5

        predicted_class = Config.class_names[label]

        true_class_id = np.unique(true_mask[y1:y2, x1:x2])
        true_class = 'background'
        for class_id in true_class_id:
            if class_id != 0:
                true_class = Config.class_names[class_id]
                break

        pred_mask[y1:y2, x1:x2][resized_mask] = label

        iou = calculate_iou(pred_mask, true_mask, label)

        iou_per_object.append({
            'image_name': image_name,
            'object_id': idx + 1,
            'predicted_class': predicted_class,
            'ground_truth_class': true_class,
            'iou': round(iou, 4)
        })

    save_iou_results(iou_per_object)

    for result in iou_per_object:
        class_name = result['predicted_class']
        all_iou_scores[class_name].append(result['iou'])
        update_class_metrics(pred_mask, true_mask, label, class_name)


def save_iou_results(results):
    csv_path = os.path.join(Config.output_dir, "iou_results_per_object.csv")
    fieldnames = ['image_name', 'object_id', 'predicted_class', 'ground_truth_class', 'iou']
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for row in results:
            writer.writerow(row)


def update_class_metrics(pred_mask, true_mask, class_id, class_name):
    pred_class = (pred_mask == class_id)
    true_class = (true_mask == class_id)

    tp = np.logical_and(pred_class, true_class).sum()
    fp = np.logical_and(pred_class, ~true_class).sum()
    fn = np.logical_and(~pred_class, true_class).sum()

    class_metrics[class_name]['true_pos'] += tp
    class_metrics[class_name]['false_pos'] += fp
    class_metrics[class_name]['false_neg'] += fn


def save_map_to_csv(map50, map50_95, class_wise_map50, class_wise_map50_95, output_dir):
    csv_path = os.path.join(output_dir, "map_results.csv")
    header = ["Metric Type", "Class", "Precision (P)", "Recall (R)", "mAP50", "mAP50-95"]

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        total_tp = sum(class_metrics[c]['true_pos'] for c in class_metrics)
        total_fp = sum(class_metrics[c]['false_pos'] for c in class_metrics)
        total_fn = sum(class_metrics[c]['false_neg'] for c in class_metrics)

        precision_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        writer.writerow([
            "All classes",
            "ALL",
            f"{precision_all:.16f}",
            f"{recall_all:.16f}",
            f"{map50:.16f}",
            f"{map50_95:.16f}"
        ])

        for class_id in range(1, len(Config.class_names)):
            class_name = Config.class_names[class_id]
            tp = class_metrics[class_name]['true_pos']
            fp = class_metrics[class_name]['false_pos']
            fn = class_metrics[class_name]['false_neg']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score50 = class_wise_map50[class_id - 1] if class_id - 1 < len(class_wise_map50) else 0.0
            score50_95 = class_wise_map50_95[class_id - 1] if class_id - 1 < len(class_wise_map50_95) else 0.0

            writer.writerow([
                "Per class",
                class_name,
                f"{precision:.16f}",
                f"{recall:.16f}",
                f"{score50:.16f}",
                f"{score50_95:.16f}"
            ])

    print(f"Результаты mAP сохранены в: {csv_path}")


def print_map_metrics(map50, map50_95, class_wise_map50, class_wise_map50_95):
    print("\nMean Average Precision Metrics:")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP50-95: {map50_95:.4f}")

    print("\nmAP-50:")
    for class_id in range(1, len(Config.class_names)):
        class_name = Config.class_names[class_id]
        score = class_wise_map50[class_id - 1] if class_id - 1 < len(class_wise_map50) else 0.0
        print(f"{class_name}: {score:.4f}")

    print("\nmAP50-95:")
    for class_id in range(1, len(Config.class_names)):
        class_name = Config.class_names[class_id]
        score = class_wise_map50_95[class_id - 1] if class_id - 1 < len(class_wise_map50_95) else 0.0
        print(f"{class_name}: {score:.4f}")


def print_final_metrics():
    print("\nИтоговые метрики:")
    for class_name in class_metrics:
        if class_name == 'background':
            continue

        tp = class_metrics[class_name]['true_pos']
        fp = class_metrics[class_name]['false_pos']
        fn = class_metrics[class_name]['false_neg']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        mean_iou = np.mean(all_iou_scores[class_name]) if all_iou_scores[class_name] else 0

        print(f"\nКласс {class_name}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")


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

    all_preds = []
    all_targets = []

    for image_file in os.listdir(Config.test_image_dir):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(Config.test_image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image)
        image_id = int(os.path.splitext(image_file)[0])

        with torch.no_grad():
            predictions = model([img_tensor.to(Config.device)])

        ground_truth = load_ground_truth(image_file, Config.test_annotations_dir)
        true_mask = load_true_mask(image_file, Config.masks_folder)
        process_image_predictions(predictions, true_mask, image_file)
        output_path = os.path.join(Config.output_dir, f"pred_{image_file}")
        save_image_with_predictions(image, predictions, output_path, ground_truth, true_mask)

        if ground_truth is not None:
            pred_coco = convert_to_coco_format(predictions, image_id, image.height, image.width)
            gt_coco = convert_ground_truth_to_coco(ground_truth, image_id, image.height, image.width)

            all_preds.append(pred_coco)
            all_targets.append(gt_coco)

    if all_preds and all_targets:
        map50, map50_95, class_wise_map50, class_wise_map50_95 = calculate_map(all_preds, all_targets)
        save_map_to_csv(map50, map50_95, class_wise_map50, class_wise_map50_95, Config.output_dir)
        print_map_metrics(map50, map50_95, class_wise_map50, class_wise_map50_95)

    print_final_metrics()


if __name__ == "__main__":
    main()
