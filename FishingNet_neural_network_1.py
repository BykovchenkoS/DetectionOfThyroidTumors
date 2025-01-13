import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import time
import os
import torchvision.models.detection as detection
from coco_for_pytorch import CustomDataset, transform, category_map
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torchvision import transforms

result_dir = 'result_fishingNet'
os.makedirs(result_dir, exist_ok=True)

weights = detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = detection.fasterrcnn_resnet50_fpn(weights=weights)

num_classes = len(category_map) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

transform_with_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(
    images_dir='dataset_coco_neuro_1/train/images',
    annotations_dir='dataset_coco_neuro_1/train/annotations',
    transforms=transform_with_augmentation
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


def collect_predictions(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            all_preds.extend([output['boxes'].cpu().numpy() for output in outputs])
            all_targets.extend([target['boxes'].cpu().numpy() for target in targets])

    return all_preds, all_targets


def compute_metrics(all_preds, all_targets):
    y_true = [1 if len(t) > 0 else 0 for t in all_targets]
    y_pred = [1 if len(p) > 0 else 0 for p in all_preds]

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)

    return cm, cm_normalized, precision, recall, f1, precisions, recalls


def plot_and_save_metrics(cm, cm_normalized, precision, recall, f1, precisions, recalls, epoch):
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{result_dir}/confusion_matrix_epoch_{epoch}.png')
    plt.close()

    # Normalized Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{result_dir}/confusion_matrix_normalized_epoch_{epoch}.png')
    plt.close()

    # Precision-Recall Curve
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f'{result_dir}/precision_recall_curve_epoch_{epoch}.png')
    plt.close()

    # F1 Score Curve
    plt.plot(range(len(f1)), f1, marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.savefig(f'{result_dir}/f1_score_curve_epoch_{epoch}.png')
    plt.close()


def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch):
    model.train()
    loss_total = 0
    start_time = time.time()

    print(f"Epoch {epoch} - Starting training...")

    for i, (images, targets, _) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_total += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch} Iteration {i}: Loss = {losses.item():.4f}")

    print(f"Epoch #{epoch} - Total Loss: {loss_total:.4f} - Time: {time.time() - start_time:.2f}s")

    all_preds, all_targets = collect_predictions(model, data_loader, device)
    cm, cm_normalized, precision, recall, f1, precisions, recalls = compute_metrics(all_preds, all_targets)

    print(f"Confusion Matrix:\n{cm}")
    print(f"Normalized Confusion Matrix:\n{cm_normalized}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    plot_and_save_metrics(cm, cm_normalized, precision, recall, f1, precisions, recalls, epoch)

    with open(f'{result_dir}/results_epoch_{epoch}.json', 'w') as f:
        json.dump({
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precisions': precisions.tolist(),
            'recalls': recalls.tolist()
        }, f)

    scheduler.step(loss_total)


num_epochs = 100
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, scheduler, train_loader, device, epoch)

model_save_path = f"{result_dir}/fishing_net_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")
