import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from detr.models.detr import PostProcess, PostProcessSegm
from detr.util.misc import collate_fn as default_collate_fn
from detr.util.misc import NestedTensor
from my_detr.DETR_neural_network import CustomDataset, get_model_instance_segmentation
import json
import cv2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Precision, Recall, F1Score
from torchvision.ops import box_iou

PREDICTIONS_DIR = "../detr_predictions"
VISUALIZATIONS_DIR = "../detr_visualizations"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


class ValidationDataset(CustomDataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        super().__init__(images_dir, annotations_dir, transforms=transforms)


def get_transform(train=False):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def visualize_masks(image, masks, labels, scores, alpha=0.5):
    """
    Визуализирует маски на изображении.
    :param image: Исходное изображение (H, W, 3).
    :param masks: Маски (N, 1, H, W).
    :param labels: Метки классов (N,).
    :param scores: Уверенность модели (N,).
    :param alpha: Прозрачность масок.
    """
    image = image.copy()
    for mask, label, score in zip(masks, labels, scores):
        # Убираем лишнюю ось у маски (N, 1, H, W) -> (H, W)
        mask = mask.squeeze(0)

        # Преобразуем маску в бинарный формат
        mask = (mask > 0.5).astype(np.uint8)  # Порог 0.5 для бинаризации

        # Генерируем случайный цвет для маски
        color = np.random.rand(3) * 255  # Случайный цвет для маски

        # Накладываем маску на изображение
        for c in range(3):  # Применяем цвет к каждому каналу
            image[:, :, c] = np.where(mask == 1, (1 - alpha) * image[:, :, c] + alpha * color[c], image[:, :, c])

    return image


def calculate_metrics(model, data_loader, device):
    num_classes = 3

    metric_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='macro')
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='macro')
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    model.eval()

    with torch.no_grad():
        for images, targets in data_loader:
            # Если images является объектом NestedTensor, извлекаем тензор изображений
            if isinstance(images, NestedTensor):
                tensor_images, _ = images.decompose()
                images = [img.to(device) for img in tensor_images]
            else:
                images = [image.to(device) for image in images]

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            preds_map = []
            targets_map = []

            for pred, target in zip(predictions, targets):
                preds_map.append({
                    "boxes": pred["boxes"].cpu(),
                    "scores": pred["scores"].cpu(),
                    "labels": pred["labels"].cpu(),
                })
                targets_map.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                })

                iou_matrix = box_iou(pred["boxes"], target["boxes"])
                max_iou, matched_gt_indices = torch.max(iou_matrix, dim=1)

                valid_preds = max_iou >= 0.5
                matched_pred_labels = pred["labels"][valid_preds].cpu()
                matched_gt_labels = target["labels"][matched_gt_indices[valid_preds]].cpu()

                if len(matched_pred_labels) > 0 and len(matched_gt_labels) > 0:
                    precision_metric.update(matched_pred_labels, matched_gt_labels)
                    recall_metric.update(matched_pred_labels, matched_gt_labels)
                    f1_metric.update(matched_pred_labels, matched_gt_labels)

            metric_map.update(preds_map, targets_map)

    result_map = metric_map.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()

    return {
        "map_50": result_map["map_50"].item() if "map_50" in result_map else 0,
        "map_50_95": result_map["map"].item() if "map" in result_map else 0,
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item()
    }


if __name__ == '__main__':
    val_dataset = ValidationDataset(
        images_dir='../dataset_coco_neuro_1/val/images',
        annotations_dir='../dataset_coco_neuro_1/val/annotations',
        transforms=get_transform(train=False)
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=default_collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 3
    model, _, postprocessors = get_model_instance_segmentation(num_classes)
    model.to(device)

    model_path = "detr_screen.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    postprocessor = PostProcess()
    postprocessor_seg = PostProcessSegm()

    image_files = sorted(os.listdir('../dataset_for_search_1/val/images'))

    for i, (images, targets) in enumerate(val_data_loader):
        if isinstance(images, NestedTensor):
            tensor_images, masks = images.decompose()
            images = [img.to(device) for img in tensor_images]
        else:
            images = list(image.to(device) for image in images)

        outputs = model(images)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        max_target_sizes = torch.stack([t["size"] for t in targets], dim=0).to(device)
        results = postprocessor(outputs, orig_target_sizes)
        results_seg = postprocessor_seg(results, outputs, orig_target_sizes, max_target_sizes)

        for j, result in enumerate(results_seg):
            image_filename = image_files[i * val_data_loader.batch_size + j]
            image_path = os.path.join('../dataset_for_search_1/val/images', image_filename)
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Конвертация в RGB

            # Получение масок, меток и уверенностей
            pred_masks = result["masks"].cpu().numpy()  # Маски (N, 1, H, W)
            pred_labels = result["labels"].cpu().numpy()
            pred_scores = result["scores"].cpu().numpy()

            # Визуализация масок на изображении
            visualized_image = visualize_masks(original_image, pred_masks, pred_labels, pred_scores)

            # Сохранение визуализированного изображения
            visualization_filename = os.path.splitext(image_filename)[0] + "_visualized.png"
            visualization_file = os.path.join(VISUALIZATIONS_DIR, visualization_filename)
            plt.imsave(visualization_file, visualized_image)

            # Сохранение предсказаний в JSON
            prediction_filename = os.path.splitext(image_filename)[0] + ".json"
            prediction_file = os.path.join(PREDICTIONS_DIR, prediction_filename)
            prediction = {
                "boxes": result["boxes"].cpu().numpy().tolist(),
                "labels": result["labels"].cpu().numpy().tolist(),
                "scores": result["scores"].cpu().numpy().tolist(),
                "masks": result["masks"].cpu().numpy().tolist()
            }
            with open(prediction_file, "w") as f:
                json.dump(prediction, f)

    # Расчет метрик
    metrics = calculate_metrics(model, val_data_loader, device)
    print("Metrics:")
    print(f"mAP@50: {metrics['map_50']}")
    print(f"mAP@50:95: {metrics['map_50_95']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1']}")

    print(f"Предсказания сохранены в папке: {PREDICTIONS_DIR}")
    print(f"Визуализации сохранены в папке: {VISUALIZATIONS_DIR}")