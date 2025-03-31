import os
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
import pandas as pd
import seaborn as sns


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'retinanet_model_node.pth'
    test_image_dir = '../dataset_coco_neuro_3/images_neuro_3'
    test_annotations_dir = '../dataset_coco_neuro_3/ann'
    output_dir = '../predict_retinanet_node'
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


def visualize_predictions(image, predictions, ground_truth=None):
    fig, ax = plt.subplots(1, 2 if ground_truth else 1, figsize=(20, 10))

    if ground_truth:
        ax[0].imshow(image)
        plot_boxes(ax[0], predictions, Config.colors['prediction'], 'Predictions')

        ax[1].imshow(image)
        plot_boxes(ax[1], ground_truth, Config.colors['ground_truth'], 'Ground Truth')
    else:
        ax.imshow(image)
        plot_boxes(ax, predictions, Config.colors['prediction'], 'Predictions')

    plt.tight_layout()
    return fig


def plot_boxes(ax, boxes_data, color, title):
    ax.set_title(title)

    if isinstance(boxes_data, dict):
        boxes = boxes_data['boxes'].cpu().numpy()
        labels = boxes_data['labels'].cpu().numpy()
        scores = boxes_data['scores'].cpu().numpy()

        keep = scores >= Config.confidence_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        for box, label, score in zip(boxes, labels, scores):
            draw_box(ax, box, Config.class_names.get(label, str(label)),
                     color, f"{score:.2f}")
    else:
        for ann in boxes_data:
            box = ann['bbox']
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            label_id = ann['category_id']
            draw_box(ax, box, Config.class_names.get(label_id, str(label_id)), color)

    ax.axis('off')


def draw_box(ax, box, label, color, score=None):
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin

    rect = plt.Rectangle(
        (xmin, ymin), width, height,
        fill=False, edgecolor=color, linewidth=2
    )
    ax.add_patch(rect)

    text = f"{label}: {score}" if score else label
    ax.text(
        xmin, ymin - 5, text,
        bbox=dict(facecolor=color, alpha=0.5),
        fontsize=8, color='white'
    )


all_true_labels = []
all_pred_labels = []
all_scores = []


def calculate_metrics(ground_truth, predictions):
    """
    Сравнивает Ground Truth с предсказаниями модели и собирает метрики.
    """
    if ground_truth is None or len(ground_truth) == 0:
        return

    gt_boxes = [ann['bbox'] for ann in ground_truth]
    gt_labels = [ann['category_id'] for ann in ground_truth]

    pred_boxes = predictions['boxes'].cpu().numpy()
    pred_labels = predictions['labels'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()

    keep = pred_scores >= Config.confidence_threshold
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    # Для каждого GT объекта (Node) проверяем, есть ли совпадение с предсказаниями
    for gt_label in gt_labels:
        # Все GT объекты - это Node (класс 1)
        all_true_labels.append(1)

        # Ищем совпадения с предсказаниями
        matched = False
        for pred_label, pred_score in zip(pred_labels, pred_scores):
            if pred_label == 1:  # Если предсказан Node
                matched = True
                all_pred_labels.append(1)
                all_scores.append(pred_score)
                break

        # Если не нашли совпадения, считаем что предсказан Background
        if not matched:
            all_pred_labels.append(2)
            all_scores.append(0)  # Уверенность 0 для Background

def plot_confusion_matrix(save_dir):
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[1, 2])
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
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[1, 2], normalize='true')
    df_cm = pd.DataFrame(cm, index=['Node', 'Background'], columns=['Node', 'Background'])

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    save_path = os.path.join(save_dir, "normalized_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_f1_precision_recall_curves(save_dir):
    precisions, recalls, thresholds = precision_recall_curve(all_true_labels, all_scores)

    # Вычисляем F1 Score для каждого порога
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

    # Построение F1 Curve
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='green')
    plt.title('F1 Score Curve')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()

    save_path = os.path.join(save_dir, "f1_score_curve.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Построение Precision Curve
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precisions[:-1], label='Precision', color='blue')
    plt.title('Precision Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.legend()

    save_path = os.path.join(save_dir, "precision_curve.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Построение Recall Curve
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, recalls[:-1], label='Recall', color='red')
    plt.title('Recall Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.legend()

    save_path = os.path.join(save_dir, "recall_curve.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_precision_recall_curve(save_dir):
    precisions, recalls, _ = precision_recall_curve(all_true_labels, all_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(recalls, precisions, label='Precision-Recall Curve', color='orange')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    save_path = os.path.join(save_dir, "precision_recall_curve.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    model = load_model()
    os.makedirs(Config.output_dir, exist_ok=True)

    for image_file in tqdm(os.listdir(Config.test_image_dir)):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(Config.test_image_dir, image_file)

        try:
            image, image_tensor = load_image(image_path)
            ground_truth = load_annotations(image_path)

            if len(ground_truth) == 0:
                print(f"Внимание: Нет Ground Truth для изображения {image_file}")
                continue

            with torch.no_grad():
                predictions = model(image_tensor)[0]

            calculate_metrics(ground_truth, predictions)

            fig = visualize_predictions(image, predictions, ground_truth)

            output_path = os.path.join(Config.output_dir, f"pred_{image_file}")
            fig.savefig(output_path, bbox_inches='tight', dpi=100)
            plt.close(fig)

        except Exception as e:
            print(f"Ошибка при обработке {image_file}: {str(e)}")


if __name__ == "__main__":
    main()
    print("Визуализация предсказаний завершена.")

    # Проверяем длины массивов
    min_length = min(len(all_true_labels), len(all_pred_labels), len(all_scores))
    if min_length == 0:
        print("Нет данных для построения метрик!")
    else:
        # Обрезаем массивы до минимальной длины (на всякий случай)
        all_true_labels = all_true_labels[:min_length]
        all_pred_labels = all_pred_labels[:min_length]
        all_scores = all_scores[:min_length]

        metrics_dir = os.path.join(Config.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        plot_confusion_matrix(metrics_dir)
        plot_normalized_confusion_matrix(metrics_dir)
        plot_f1_precision_recall_curves(metrics_dir)
        plot_precision_recall_curve(metrics_dir)
        print(f"Графики сохранены в директорию: {metrics_dir}")
