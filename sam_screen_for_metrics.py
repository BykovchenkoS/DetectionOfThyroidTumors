import os
import json
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry
from tqdm import tqdm
import csv
from sklearn.metrics import average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F

CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
FINE_TUNED_MODEL_PATH = "sam_best_screen.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INFERENCE_IMAGES_DIR = "dataset_sam_neuro_1/val/images"
ANNOTATIONS_DIR = "dataset_sam_neuro_1/val/annotations"
OUTPUT_DIR = "sam_predictions_screen_val_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['Thyroid tissue', 'Carotis']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLASS_IDS = list(CLASS_TO_ID.values())

CSV_METRICS_PATH = os.path.join(OUTPUT_DIR, "inference_metrics.csv")
CSV_PER_OBJECT_PATH = os.path.join(OUTPUT_DIR, "per_object_metrics.csv")
CSV_MAP_PATH = os.path.join(OUTPUT_DIR, "image_mAP.csv")


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image.astype(np.uint8)


def load_annotations(annotation_dir, image_name):
    base_name = os.path.splitext(image_name)[0]
    ann_path = os.path.join(annotation_dir, f"{base_name}.json")
    if not os.path.exists(ann_path):
        return None
    with open(ann_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Invalid JSON: {ann_path}")
            return None
    annotations = []
    category_map = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    for obj in data.get("annotations", []):
        if 'bbox' not in obj or 'category_id' not in obj:
            continue
        bbox = obj["bbox"]
        class_name = category_map.get(obj["category_id"], "Unknown")
        if class_name not in CLASS_TO_ID:
            continue
        class_id = CLASS_TO_ID[class_name]
        if len(bbox) != 4:
            continue
        annotations.append({
            "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
            "class_id": class_id,
            "segmentation": obj.get("segmentation", [])
        })
    return annotations


def create_mask_from_segmentation(segmentation, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, list):
        try:
            poly = np.array(segmentation).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
        except Exception as e:
            print(f"Segmentation processing error: {e}")
    elif isinstance(segmentation, str):
        try:
            mask = cv2.imread(segmentation, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Failed to load mask")
            mask = (mask > 0).astype(np.uint8)
        except Exception as e:
            print(f"Mask loading error: {e}")
    return mask


def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / (union + 1e-8)


def calculate_object_metrics(pred_mask, true_mask):
    pred_bin = (pred_mask > 0.5).astype(np.float32)
    true_bin = true_mask.astype(np.float32)
    tp = np.sum(pred_bin * true_bin)
    fp = np.sum(pred_bin * (1 - true_bin))
    fn = np.sum((1 - pred_bin) * true_bin)
    metrics = {
        "accuracy": (tp + np.sum((1 - pred_bin) * (1 - true_bin))) / (true_mask.size + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "iou": calculate_iou(pred_bin, true_bin)
    }
    return {k: float(v) for k, v in metrics.items()}


def calculate_class_map(all_true_masks, all_pred_scores):
    if not all_true_masks:
        return {"map50": 0.0, "map95": 0.0, "map": 0.0}
    y_true = np.concatenate([mask.flatten() for mask in all_true_masks])
    y_pred = np.concatenate([score.flatten() for score in all_pred_scores])
    thresholds = np.linspace(0.5, 0.95, 10)
    aps = []
    for threshold in thresholds:
        y_pred_thresh = (y_pred > threshold).astype(np.float32)
        ap = average_precision_score(y_true, y_pred_thresh)
        aps.append(ap)
    return {
        "map50": float(aps[0]) if aps else 0.0,
        "map95": float(aps[-1]) if aps else 0.0,
        "map": float(np.mean(aps)) if aps else 0.0
    }


def save_metrics_to_csv(metrics_dict, filename, fieldnames):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)


def visualize_results(original_image, pred_masks, true_masks, bboxes, classes, output_path):
    pred_vis = original_image.copy()
    gt_vis = original_image.copy()
    colors = [(255, 0, 255), (0, 255, 0)]

    for mask, bbox, cls_id in zip(pred_masks, bboxes, classes):
        color = colors[cls_id]
        label = ID_TO_CLASS[cls_id]
        mask = mask.astype(bool)
        pred_vis[mask] = (0.6 * np.array(color) + 0.4 * pred_vis[mask]).astype(np.uint8)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(pred_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(pred_vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for mask, bbox, cls_id in zip(true_masks, bboxes, classes):
        color = colors[cls_id]
        label = ID_TO_CLASS[cls_id]
        mask = mask.astype(bool)
        gt_vis[mask] = (0.6 * np.array(color) + 0.4 * gt_vis[mask]).astype(np.uint8)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(gt_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(gt_vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(pred_vis, cv2.COLOR_BGR2RGB))
    ax1.set_title("Predictions")
    ax1.axis('off')
    ax2.imshow(cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB))
    ax2.set_title("Ground Truth")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(DEVICE)
    try:
        checkpoint = torch.load(FINE_TUNED_MODEL_PATH, map_location=DEVICE)
        sam.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
    except Exception as e:
        print(f"Model loading error: {e}")
        return
    sam.eval()

    class_true_masks = {cls_id: [] for cls_id in CLASS_IDS}
    class_pred_scores = {cls_id: [] for cls_id in CLASS_IDS}
    per_object_results = []
    all_gt_classes = []
    all_pred_classes = []

    image_files = [f for f in os.listdir(INFERENCE_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
    image_map_data = []

    for img_name in tqdm(image_files, desc="Processing images"):
        try:
            img_path = os.path.join(INFERENCE_IMAGES_DIR, img_name)
            original_image = load_image(img_path)
            height, width = original_image.shape[:2]
            annotations = load_annotations(ANNOTATIONS_DIR, img_name)
            if not annotations:
                continue

            image_true_masks = {cls_id: [] for cls_id in CLASS_IDS}
            image_pred_scores = {cls_id: [] for cls_id in CLASS_IDS}

            pred_masks = []
            true_masks = []
            bboxes = []
            classes = []

            for obj_idx, ann in enumerate(annotations):
                try:
                    true_mask = create_mask_from_segmentation(ann["segmentation"], height, width)
                    if true_mask.sum() == 0:
                        continue

                    box_tensor = torch.tensor(ann["bbox"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    image_tensor = torch.from_numpy(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
                    image_tensor = image_tensor / 255.0
                    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        image_embedding = sam.image_encoder(image_tensor)
                        sparse_emb, dense_emb = sam.prompt_encoder(
                            points=None,
                            boxes=box_tensor,
                            masks=None
                        )

                        low_res_mask, _ = sam.mask_decoder(
                            image_embeddings=image_embedding,
                            image_pe=sam.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False,
                        )

                        pred_mask = F.interpolate(
                            low_res_mask,
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze().cpu().numpy()

                    class_id = ann["class_id"]
                    class_true_masks[class_id].append(true_mask)
                    class_pred_scores[class_id].append(pred_mask)
                    image_true_masks[class_id].append(true_mask)
                    image_pred_scores[class_id].append(pred_mask)

                    obj_metrics = calculate_object_metrics(pred_mask, true_mask)
                    per_object_results.append({
                        'image_name': img_name,
                        'obj_idx': obj_idx,
                        'class_id': class_id,
                        **obj_metrics
                    })

                    pred_masks.append((pred_mask > 0.5).astype(np.uint8))
                    true_masks.append(true_mask)
                    bboxes.append(ann["bbox"])
                    classes.append(class_id)

                    all_gt_classes.append(class_id)
                    all_pred_classes.append(class_id)

                except Exception as e:
                    print(f"Object {obj_idx} in {img_name} error: {e}")
                    continue

            vis_path = os.path.join(OUTPUT_DIR, f"vis_{os.path.splitext(img_name)[0]}.png")
            visualize_results(original_image, pred_masks, true_masks, bboxes, classes, vis_path)

            image_map_metrics = {}
            for cls_id in CLASS_IDS:
                map_metrics = calculate_class_map(image_true_masks[cls_id], image_pred_scores[cls_id])
                image_map_metrics[f"map50_{ID_TO_CLASS[cls_id]}"] = map_metrics["map50"]
                image_map_metrics[f"map95_{ID_TO_CLASS[cls_id]}"] = map_metrics["map95"]

            image_map_data.append({
                'image_name': img_name,
                **image_map_metrics
            })

        except Exception as e:
            print(f"Image {img_name} processing error: {e}")
            continue

    fieldnames = ['image_name'] + [f"map50_{cls}" for cls in CLASS_NAMES] + [f"map95_{cls}" for cls in CLASS_NAMES]
    save_metrics_to_csv({}, CSV_MAP_PATH, fieldnames)
    for data in image_map_data:
        save_metrics_to_csv(data, CSV_MAP_PATH, fieldnames)

    class_metrics = {}
    for cls_id in CLASS_IDS:
        map_metrics = calculate_class_map(class_true_masks[cls_id], class_pred_scores[cls_id])
        cls_objects = [obj for obj in per_object_results if obj['class_id'] == cls_id]
        if cls_objects:
            avg_metrics = {
                'accuracy': np.mean([obj['accuracy'] for obj in cls_objects]),
                'precision': np.mean([obj['precision'] for obj in cls_objects]),
                'recall': np.mean([obj['recall'] for obj in cls_objects]),
                'iou': np.mean([obj['iou'] for obj in cls_objects]),
                **map_metrics
            }
        else:
            avg_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'iou': 0.0, **map_metrics}
        class_metrics[cls_id] = avg_metrics

    for cls_id, metrics in class_metrics.items():
        save_metrics_to_csv(
            {
                'epoch': 1,
                'phase': 'test',
                'class_id': cls_id,
                'class_name': ID_TO_CLASS[cls_id],
                **metrics
            },
            CSV_METRICS_PATH,
            ['epoch', 'phase', 'class_id', 'class_name', 'accuracy', 'precision', 'recall', 'iou', 'map50', 'map95', 'map']
        )

    for obj in per_object_results:
        save_metrics_to_csv(
            {
                'epoch': 1,
                'phase': 'test',
                **obj,
                'class_name': ID_TO_CLASS[obj['class_id']]
            },
            CSV_PER_OBJECT_PATH,
            ['epoch', 'phase', 'image_name', 'obj_idx', 'class_id', 'class_name', 'accuracy', 'precision', 'recall', 'iou']
        )

    if all_gt_classes:
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        cm = confusion_matrix(all_gt_classes, all_pred_classes, labels=CLASS_IDS)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(cm_path)
        plt.close()

    print("\nEvaluation complete!")
    print(f"Class metrics saved to: {CSV_METRICS_PATH}")
    print(f"Per-object metrics saved to: {CSV_PER_OBJECT_PATH}")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print(f"mAP per image saved to: {CSV_MAP_PATH}")


if __name__ == "__main__":
    main()
