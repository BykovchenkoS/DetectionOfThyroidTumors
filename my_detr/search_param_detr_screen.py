import optuna
from optuna.trial import TrialState
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import logging
import torch
from my_detr.DETR_neural_network import CustomDataset, get_model_instance_segmentation
from detr.util.misc import collate_fn as default_collate_fn
import torchvision.transforms as T
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import Precision, Recall, F1Score
from torch.utils.data import DataLoader
from detr.util.misc import NestedTensor
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detr_hyperparameter_tuning.log"),
        logging.StreamHandler()
    ]
)


def calculate_metrics(model, data_loader, device):
    num_classes = 3
    metric_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='macro')
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='macro')
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            # Проверка на тип NestedTensor и извлечение тензоров
            if isinstance(images, NestedTensor):
                tensor_images, _ = images.decompose()
                images = [img.to(device) for img in tensor_images]
            else:
                images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            preds_map = []
            targets_map = []
            for i in range(len(outputs["pred_logits"])):
                logits = outputs["pred_logits"][i].softmax(-1)
                scores, labels = logits.max(-1)
                boxes = outputs["pred_boxes"][i]

                keep = scores > 0.5
                pred_boxes = boxes[keep].cpu()
                pred_scores = scores[keep].cpu()
                pred_labels = labels[keep].cpu()

                preds_map.append({
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                })
                targets_map.append({
                    "boxes": targets[i]["boxes"].cpu(),
                    "labels": targets[i]["labels"].cpu(),
                })

                if len(pred_boxes) > 0 and len(targets[i]["boxes"]) > 0:
                    iou_matrix = box_iou(pred_boxes, targets[i]["boxes"])
                    max_iou, matched_gt_indices = torch.max(iou_matrix, dim=1)
                    valid_preds = max_iou >= 0.5
                    matched_pred_labels = pred_labels[valid_preds]
                    matched_gt_labels = targets[i]["labels"][matched_gt_indices[valid_preds]].cpu()

                    if len(matched_pred_labels) > 0 and len(matched_gt_labels) > 0:
                        precision_metric.update(matched_pred_labels, matched_gt_labels)
                        recall_metric.update(matched_pred_labels, matched_gt_labels)
                        f1_metric.update(matched_pred_labels, matched_gt_labels)

        metric_map.update(preds_map, targets_map)

    result_map = metric_map.compute()
    precision = precision_metric.compute() if precision_metric._update_count > 0 else torch.tensor(0.0)
    recall = recall_metric.compute() if recall_metric._update_count > 0 else torch.tensor(0.0)
    f1 = f1_metric.compute() if f1_metric._update_count > 0 else torch.tensor(0.0)

    return {
        "map_50": result_map["map_50"].item() if "map_50" in result_map else 0,
        "map_50_95": result_map["map"].item() if "map" in result_map else 0,
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item()
    }


def get_transform(train=False):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def objective(trial):
    # Подбор гиперпараметров
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    num_epochs = trial.suggest_int("num_epochs", 10, 50)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])

    # Логирование параметров
    logging.info(f"Trial {trial.number}: Parameters - "
                 f"batch_size={batch_size}, "
                 f"learning_rate={learning_rate:.2e}, "
                 f"weight_decay={weight_decay:.2e}, "
                 f"num_epochs={num_epochs}, "
                 f"optimizer={optimizer_name}, "
                 f"scheduler={scheduler_name}")

    num_classes = 3
    model, _, _ = get_model_instance_segmentation(num_classes)
    matcher = HungarianMatcher()
    weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
    losses = ["labels", "boxes", "cardinality"]
    eos_coef = 0.1
    criterion = SetCriterion(
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses
    )
    model.criterion = criterion

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_dataset = CustomDataset(
        images_dir='../dataset_for_search_1/train/images',
        annotations_dir='../dataset_for_search_1/train/annotations',
        transforms=get_transform(train=True)
    )
    val_dataset = CustomDataset(
        images_dir='../dataset_for_search_1/val/images',
        annotations_dir='../dataset_for_search_1/val/annotations',
        transforms=get_transform(train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=default_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=default_collate_fn
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    if scheduler_name == "StepLR":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1)

    best_f1_score = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            # Проверка на тип NestedTensor и извлечение тензоров
            if isinstance(images, NestedTensor):
                tensor_images, _ = images.decompose()
                images = [img.to(device) for img in tensor_images]
            else:
                images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()

        # Валидация
        val_metrics = calculate_metrics(model, val_loader, device)
        f1_score = val_metrics['f1']
        logging.info(f"Trial {trial.number} - Epoch {epoch + 1}/{num_epochs}: "
                     f"Loss = {epoch_loss:.4f}, "
                     f"F1-score = {f1_score:.4f}, "
                     f"Precision = {val_metrics['precision']:.4f}, "
                     f"Recall = {val_metrics['recall']:.4f}, "
                     f"mAP50 = {val_metrics['map_50']:.4f}")

        # Сохранение лучшей модели
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            torch.save(model.state_dict(), f"best_model_trial_{trial.number}_detr.pth")

        # Обновление планировщика
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(f1_score)
        else:
            scheduler.step()

    return best_f1_score


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, timeout=600)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")