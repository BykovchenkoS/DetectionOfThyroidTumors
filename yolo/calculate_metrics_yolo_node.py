import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import os
import csv

weights_path = 'YOLO_result/train4_node_yolo5/weights/best.pt'
model = YOLO(weights_path)

images_path = 'dataset_yolo_neuro_2/images/val'
image_files = list(Path(images_path).glob('*.jpg'))

results_dir = 'predict_yolo_5_node'
os.makedirs(results_dir, exist_ok=True)

iou_csv_path = Path(results_dir) / 'iou_results_per_object.csv'
map_csv_path = Path(results_dir) / 'map_results.csv'


def parse_yolo_label(label_path, img_width, img_height):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            labels.append((cls, x1, y1, x2, y2))
    return labels


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


def plot_boxes(ax, boxes, color, title, show_confidence=False):
    for box in boxes:
        if show_confidence:
            cls, x1, y1, x2, y2, conf = box
            label = f"{model.names[cls]} {conf:.2f}"
        else:
            cls, x1, y1, x2, y2 = box
            label = model.names[cls]

        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        ax.text(x1, y1 - 10, label, fontsize=12, color=color, bbox=dict(facecolor='white', alpha=0.5))

    ax.set_title(title)
    ax.axis('off')


def visualize_predictions(image, predictions, ground_truth=None):
    fig, axes = plt.subplots(1, 2 if ground_truth else 1, figsize=(20, 10))

    if ground_truth:
        ax_pred = axes[0]
        ax_pred.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plot_boxes(ax_pred, predictions, 'green', 'Predictions', show_confidence=True)

        ax_gt = axes[1]
        ax_gt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plot_boxes(ax_gt, ground_truth, 'red', 'Ground Truth')

    else:
        ax = axes
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plot_boxes(ax, predictions, 'green', 'Predictions', show_confidence=True)

    plt.tight_layout()
    return fig


with open(iou_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['image_name', 'object_id', 'predicted_class', 'ground_truth_class', 'iou'])

    with open(map_csv_path, mode='w', newline='') as map_file:
        map_writer = csv.writer(map_file)
        map_writer.writerow(['metric', 'value'])
        all_ground_truths = []
        all_predictions = []

        for image_path in image_files:
            img = cv2.imread(str(image_path))
            img_height, img_width, _ = img.shape
            results = model.predict(source=img)

            predictions = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    predictions.append((cls, x1, y1, x2, y2, conf))

            filtered_predictions = []
            max_confidences = {}

            for pred in predictions:
                cls, x1, y1, x2, y2, conf = pred
                if cls not in max_confidences or conf > max_confidences[cls]:
                    max_confidences[cls] = conf
                    filtered_predictions = [pred]
                elif conf == max_confidences[cls]:
                    filtered_predictions.append(pred)

            label_path = Path(f'dataset_yolo_neuro_2/labels/val/{image_path.stem}.txt')

            ground_truth = None
            if label_path.exists():
                ground_truth = parse_yolo_label(label_path, img_width, img_height)

            object_id = 1
            if ground_truth:
                for pred in filtered_predictions:
                    pred_cls, x1_pred, y1_pred, x2_pred, y2_pred, _ = pred
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
                        writer.writerow([image_path.name, object_id, model.names[pred_cls], model.names[gt_cls], best_iou])
                        object_id += 1

            if ground_truth:
                all_ground_truths.extend([(cls, x1, y1, x2, y2) for cls, x1, y1, x2, y2 in ground_truth])
            all_predictions.extend([(cls, x1, y1, x2, y2, conf) for cls, x1, y1, x2, y2, conf in filtered_predictions])

            fig = visualize_predictions(img, filtered_predictions, ground_truth)

            output_path = Path(results_dir) / f"{image_path.stem}_result.png"
            fig.savefig(str(output_path), bbox_inches='tight', pad_inches=0.5)

            plt.close(fig)

        metrics = model.val(data='YOLO_result/train4_node_yolo5/data_for_yolo_2.yaml', plots=False, verbose=False)
        map50 = metrics.box.map50
        map95 = metrics.box.map

        map_writer.writerow(['mAP50', map50])
        map_writer.writerow(['mAP95', map95])

        print(f"mAP50: {map50:.4f}")
        print(f"mAP95: {map95:.4f}")
