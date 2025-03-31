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
    model_path = 'retinanet_model_screen.pth'
    test_image_dir = '../dataset_coco_neuro_1/images_neuro_1'
    test_annotations_dir = '../dataset_coco_neuro_1/ann_neuro_1'
    output_dir = '../predict_retinanet_screen'
    confidence_threshold = 0.4
    class_names = {
        1: 'Thyroid tissue',
        2: 'Carotis',
        3: 'Background'
    }
    colors = {
        'prediction': {
            1: 'purple',
            2: 'pink',
            3: 'orange'
        },
        'ground_truth': {
            1: 'green',
            2: 'red',
            3: 'darkorange'
        }
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


def plot_boxes(ax, boxes_data, color_dict, title):
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
            class_name = Config.class_names.get(label, str(label))
            color = color_dict.get(label, 'gray')
            draw_box(ax, box, class_name, color, f"{score:.2f}")
    else:
        for ann in boxes_data:
            box = ann['bbox']
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            label_id = ann['category_id']
            class_name = Config.class_names.get(label_id, str(label_id))
            color = color_dict.get(label_id, 'gray')
            draw_box(ax, box, class_name, color)

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

    # Для каждого GT объекта проверяем, есть ли совпадение с предсказаниями
    for gt_label in gt_labels:
        all_true_labels.append(gt_label)

        # Ищем совпадения с предсказаниями
        matched = False
        for pred_label, pred_score in zip(pred_labels, pred_scores):
            if pred_label == gt_label:  # Если класс совпадает
                matched = True
                all_pred_labels.append(pred_label)
                all_scores.append(pred_score)
                break

        # Если не нашли совпадения, считаем что предсказан Background
        if not matched:
            all_pred_labels.append(3)  # 3 - это Background
            all_scores.append(0)  # Уверенность 0 для Background


def plot_confusion_matrix(save_dir):
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[1, 2, 3])
    df_cm = pd.DataFrame(cm,
                         index=['Thyroid tissue', 'Carotis', 'Background'],
                         columns=['Thyroid tissue', 'Carotis', 'Background'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_normalized_confusion_matrix(save_dir):
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[1, 2, 3], normalize='true')
    df_cm = pd.DataFrame(cm,
                         index=['Thyroid tissue', 'Carotis', 'Background'],
                         columns=['Thyroid tissue', 'Carotis', 'Background'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    save_path = os.path.join(save_dir, "normalized_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_f1_precision_recall_curves(save_dir):
    # Для многоклассовой задачи мы можем построить кривые для каждого класса
    for class_id, class_name in Config.class_names.items():
        if class_id == 3:  # Пропускаем Background
            continue

        # Создаем бинарные метки для текущего класса
        binary_true = [1 if x == class_id else 0 for x in all_true_labels]
        binary_scores = [s if p == class_id else 0 for p, s in zip(all_pred_labels, all_scores)]

        precisions, recalls, thresholds = precision_recall_curve(binary_true, binary_scores)

        # Вычисляем F1 Score для каждого порога
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

        # Построение кривых
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(thresholds, f1_scores, label=f'F1 Score ({class_name})', color='green')
        plt.title(f'F1 Score Curve ({class_name})')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(thresholds, precisions[:-1], label=f'Precision ({class_name})', color='blue')
        plt.title(f'Precision Curve ({class_name})')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(thresholds, recalls[:-1], label=f'Recall ({class_name})', color='red')
        plt.title(f'Recall Curve ({class_name})')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"metrics_curves_{class_name.lower().replace(' ', '_')}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def plot_precision_recall_curve(save_dir):
    # Для многоклассовой задачи строим кривые для каждого класса
    for class_id, class_name in Config.class_names.items():
        if class_id == 3:  # Пропускаем Background
            continue

        binary_true = [1 if x == class_id else 0 for x in all_true_labels]
        binary_scores = [s if p == class_id else 0 for p, s in zip(all_pred_labels, all_scores)]

        precisions, recalls, _ = precision_recall_curve(binary_true, binary_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, label=f'{class_name}', color=Config.colors['prediction'][class_id])
        plt.title(f'Precision-Recall Curve ({class_name})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()

        save_path = os.path.join(save_dir, f"precision_recall_curve_{class_name.lower().replace(' ', '_')}.png")
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