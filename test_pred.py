from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from torchmetrics.classification import Precision, Recall, F1Score

preds = [
    {
        "boxes": torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([1]),
    }
]

targets = [
    {
        "boxes": torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
        "labels": torch.tensor([1]),
    }
]

metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
metric.update(preds, targets)
result = metric.compute()

print(result)
print("mAP50:", result["map_50"].item())
print("mAP50-95:", result["map"].item())


pred_labels = torch.cat([pred["labels"] for pred in preds])
target_labels = torch.cat([target["labels"] for target in targets])

precision_metric = Precision(task="multiclass", num_classes=2, average='macro')
recall_metric = Recall(task="multiclass", num_classes=2, average='macro')
f1_metric = F1Score(task="multiclass", num_classes=2, average='macro')

precision = precision_metric(pred_labels, target_labels)
recall = recall_metric(pred_labels, target_labels)
f1 = f1_metric(pred_labels, target_labels)

print("\nClassification Metrics:")
print(f"Precision: {precision.item():.4f}")
print(f"Recall: {recall.item():.4f}")
print(f"F1 Score: {f1.item():.4f}")