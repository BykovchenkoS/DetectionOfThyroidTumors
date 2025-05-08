import os
import json
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry
from tqdm import tqdm
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score
import matplotlib.pyplot as plt
import torch.nn.functional as F

CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
FINE_TUNED_MODEL_PATH = "sam_best_node.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INFERENCE_IMAGES_DIR = "dataset_sam_neuro_2/val/images"
ANNOTATIONS_DIR = "dataset_sam_neuro_2/val/annotations"
OUTPUT_DIR = "sam_predictions_node_val_metrics_new"

CSV_METRICS_PATH = os.path.join(OUTPUT_DIR, "inference_metrics.csv")
CSV_PER_OBJECT_PATH = os.path.join(OUTPUT_DIR, "per_object_metrics.csv")

CLASS_NAMES = ['Node']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLASS_IDS = list(CLASS_TO_ID.values())

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_annotations(annotation_dir, image_name):
    base_name = os.path.splitext(image_name)[0]
    ann_path = os.path.join(annotation_dir, f"{base_name}.json")
    if not os.path.exists(ann_path):
        return None
    with open(ann_path) as f:
        data = json.load(f)

    annotations = []
    category_map = {cat['id']: cat['name'] for cat in data['categories']}
    for obj in data["annotations"]:
        bbox = obj["bbox"]
        class_name = category_map[obj["category_id"]]
        class_id = CLASS_TO_ID[class_name]
        annotations.append({
            "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
            "class_id": class_id,
            "segmentation": obj["segmentation"]
        })
    return annotations


def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / (union + 1e-8)


def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    pred_bin = (pred_mask > threshold).astype(np.float32)
    true_mask = true_mask.astype(np.float32)

    tp = (pred_bin * true_mask).sum()
    fp = (pred_bin * (1 - true_mask)).sum()
    fn = ((1 - pred_bin) * true_mask).sum()
    tn = ((1 - pred_bin) * (1 - true_mask)).sum()

    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = calculate_iou(pred_bin, true_mask)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou)
    }


def calculate_map(pred_mask, true_mask, thresholds=np.linspace(0.5, 0.95, 10)):
    precisions = []
    recalls = []
    for t in thresholds:
        pred_bin = (pred_mask > t).astype(np.float32)
        tp = (pred_bin * true_mask).sum()
        fp = (pred_bin * (1 - true_mask)).sum()
        fn = ((1 - pred_bin) * true_mask).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
    map = 0.0
    for r in np.linspace(0, 1, 11):
        prec_at_recall = [p for (p, rc) in zip(precisions, recalls) if rc >= r]
        if len(prec_at_recall) > 0:
            map += max(prec_at_recall) / 11

    map50 = precisions[0]
    map95 = precisions[-1]

    return {
        "map50": float(map50),
        "map95": float(map95),
        "map": float(map)
    }


def save_epoch_metrics(epoch, phase, metrics_dict, filename=CSV_METRICS_PATH):
    fieldnames = ['epoch', 'phase', 'class_id', 'class_name', 'accuracy', 'precision', 'recall', 'iou', 'map50', 'map95',
                  'map']
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for class_id, metrics in metrics_dict.items():
            row = {
                'epoch': epoch,
                'phase': phase,
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


def save_per_object_metrics(epoch, phase, object_list, filename=CSV_PER_OBJECT_PATH):
    fieldnames = ['epoch', 'phase', 'image_name', 'obj_idx', 'class_id', 'class_name',
                  'accuracy', 'precision', 'recall', 'iou', 'map50', 'map95', 'map']
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for item in object_list:
            item['epoch'] = epoch
            item['phase'] = phase
            item['class_name'] = ID_TO_CLASS[item['class_id']]
            writer.writerow(item)


def draw_confusion_matrix(y_true, y_pred, output_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.close()


def visualize_masks(original_image, pred_masks, true_masks, bboxes, classes, output_path):
    pred_vis = original_image.copy()
    gt_vis = original_image.copy()

    colors = [(255, 0, 255)]

    for mask, bbox, cls_id in zip(pred_masks, bboxes, classes):
        color = colors[cls_id]
        label = ID_TO_CLASS[cls_id]

        mask = mask.astype(bool)
        pred_vis[mask] = (0.6 * np.array(color) + 0.4 * pred_vis[mask]).astype(np.uint8)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(pred_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(pred_vis, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for mask, bbox, cls_id in zip(true_masks, bboxes, classes):
        color = colors[cls_id]
        label = ID_TO_CLASS[cls_id]

        mask = mask.astype(bool)
        gt_vis[mask] = (0.6 * np.array(color) + 0.4 * gt_vis[mask]).astype(np.uint8)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(gt_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(gt_vis, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))
    axes[0].set_title("Predictions")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(DEVICE)

checkpoint = torch.load(FINE_TUNED_MODEL_PATH, map_location=DEVICE)
sam.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
sam.eval()

metrics_per_class = {cls_id: {
    'accuracy': [], 'precision': [], 'recall': [], 'iou': [],
    'map50': [], 'map95': [], 'map': [], 'count': 0
} for cls_id in CLASS_IDS}

all_gt_classes = []
all_pred_classes = []
per_object_list = []

image_files = [f for f in os.listdir(INFERENCE_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

for img_idx, img_name in enumerate(tqdm(image_files, desc="Inference")):
    try:
        img_path = os.path.join(INFERENCE_IMAGES_DIR, img_name)
        original_image = load_image(img_path)
        height, width = original_image.shape[:2]
        annotations = load_annotations(ANNOTATIONS_DIR, img_name)
        if not annotations:
            continue

        pred_masks = []
        true_masks = []
        bboxes = []
        classes = []

        for obj_idx, ann in enumerate(annotations):
            bbox = ann["bbox"]
            true_mask = np.zeros((height, width), dtype=np.uint8)
            segmentation = ann["segmentation"]
            class_id = ann["class_id"]

            if isinstance(segmentation, list):
                poly = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(true_mask, [poly], 1)
            elif isinstance(segmentation, str):
                mask_path = os.path.join("dataset_sam_neuro_222/masks", os.path.basename(segmentation))
                true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                true_mask = (true_mask > 0).astype(np.uint8)
            else:
                raise ValueError(f"Unknown segmentation type: {type(segmentation)}")

            box_torch = torch.tensor(bbox, dtype=torch.float, device=DEVICE).unsqueeze(0)
            image_tensor = torch.tensor(original_image).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                image_embeddings = sam.image_encoder(image_tensor)

                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )

                pred_mask = F.interpolate(
                    low_res_masks,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().cpu().numpy()

            metrics = calculate_metrics(pred_mask, true_mask)
            map_metrics = calculate_map(pred_mask, true_mask)

            per_object_list.append({
                'image_name': img_name,
                'obj_idx': obj_idx,
                'class_id': class_id,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'iou': metrics['iou'],
                'map50': map_metrics['map50'],
                'map95': map_metrics['map95'],
                'map': map_metrics['map']
            })

            metrics_per_class[class_id]['accuracy'].append(metrics['accuracy'])
            metrics_per_class[class_id]['precision'].append(metrics['precision'])
            metrics_per_class[class_id]['recall'].append(metrics['recall'])
            metrics_per_class[class_id]['iou'].append(metrics['iou'])
            metrics_per_class[class_id]['map50'].append(map_metrics['map50'])
            metrics_per_class[class_id]['map95'].append(map_metrics['map95'])
            metrics_per_class[class_id]['map'].append(map_metrics['map'])
            metrics_per_class[class_id]['count'] += 1

            all_gt_classes.append(class_id)
            pred_class = class_id
            all_pred_classes.append(pred_class)

            pred_masks.append((pred_mask > 0.5).astype(np.uint8))
            true_masks.append(true_mask)
            bboxes.append(bbox)
            classes.append(class_id)

        vis_output_path = os.path.join(OUTPUT_DIR, f"vis_{os.path.splitext(img_name)[0]}.png")
        visualize_masks(original_image, pred_masks, true_masks, bboxes, classes, vis_output_path)

    except Exception as e:
        print(f"\nОшибка при обработке {img_name}: {str(e)}")
        continue

final_metrics = {}
for cls_id in CLASS_IDS:
    count = metrics_per_class[cls_id]['count']
    if count == 0:
        continue

    final_metrics[cls_id] = {
        'accuracy': np.mean(metrics_per_class[cls_id]['accuracy']),
        'precision': np.mean(metrics_per_class[cls_id]['precision']),
        'recall': np.mean(metrics_per_class[cls_id]['recall']),
        'iou': np.mean(metrics_per_class[cls_id]['iou']),
        'map50': np.mean(metrics_per_class[cls_id]['map50']),
        'map95': np.mean(metrics_per_class[cls_id]['map95']),
        'map': np.mean(metrics_per_class[cls_id]['map'])
    }

save_epoch_metrics(1, "test", final_metrics)
save_per_object_metrics(1, "test", per_object_list)

draw_confusion_matrix(all_gt_classes, all_pred_classes, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

print(f"\nМетрики по классам сохранены в: {CSV_METRICS_PATH}")
print(f"Метрики по каждому объекту сохранены в: {CSV_PER_OBJECT_PATH}")
print(f"Матрица ошибок сохранена в: {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")
print(f"Визуализации сохранены в: {OUTPUT_DIR}")
