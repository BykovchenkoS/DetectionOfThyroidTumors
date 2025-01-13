import os
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.optim as optim
import coco_for_pytorch

transform = T.Compose([T.ToTensor()])
dataset = coco_for_pytorch.CustomDataset(images_dir='dataset_coco_neuro_1/train/images',
                        annotations_dir='dataset_coco_neuro_1/train/annotations',
                        transforms=transform)

# проверка загрузки первого элемента
img, target, category_map = dataset[0]
print("Image shape:", img.shape)
print("Annotations:", target)


def visualize(img, target, category_map):
    img = img.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    # Визуализация боксов
    boxes = target['boxes']
    labels = target['labels']
    masks = target['masks']
    for box, label, mask in zip(boxes, labels, masks):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Добавление подписи класса
        category_name = category_map[label.item()]  # получаем название класса
        ax.text(xmin, ymin - 5, category_name, color='r', fontsize=10, fontweight='bold')

        # Визуализация маски
        mask = mask.cpu().numpy()
        ax.imshow(mask, alpha=0.5)

    plt.show()


model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

num_classes = 4  # 1 класс фона + 3 класса объектов

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Перевод модели на GPU (если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=1e-4)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

model.train()
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    running_loss = 0.0
    for i, (images, targets, category_map) in enumerate(train_loader):
        # Переводим изображения и аннотации на GPU
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()

        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i}/{len(train_loader)}, Loss: {losses.item()}")

    lr_scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    plt.plot(running_loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss plot - Epoch {epoch+1}")
    plt.show()

torch.save(model.state_dict(), 'mask_rcnn_model.pth')
