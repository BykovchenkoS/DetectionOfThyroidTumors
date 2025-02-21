import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics.models.sam.build import build_sam
from ultralytics.models.sam.predict import Predictor


class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_filename = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертируем в RGB
        ann_filename = img_filename.replace('.jpg', '.json')
        ann_path = os.path.join(self.annotations_dir, ann_filename)
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        boxes, labels, masks = [], [], []
        img_height, img_width = img.shape[:2]

        for ann in annotation['annotations']:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])

            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            if isinstance(ann['segmentation'], str):
                mask = np.array(Image.open(ann['segmentation']).convert("L"))
                mask = (mask > 0).astype(np.uint8)
            elif isinstance(ann['segmentation'], list):
                polygons = np.array(ann['segmentation'], dtype=np.int32)
                cv2.fillPoly(mask, [polygons], 1)
            masks.append(mask)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        masks = np.array(masks, dtype=np.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": np.array([index])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


def visualize_predictions(image, target, predictor):
    # Преобразуем изображение в формат, подходящий для SAM
    image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    predictor.set_image(image_resized)

    boxes = target["boxes"]
    masks_gt = target["masks"]

    valid_indices = target["labels"] != -1
    boxes = boxes[valid_indices]
    masks_gt = masks_gt[valid_indices]

    if len(boxes) == 0:
        print("No objects to predict on this image.")
        return

    # Выполняем инференс
    masks_pred, _, _ = predictor.inference(
        im=image_resized,
        bboxes=boxes,
        multimask_output=False
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Отображаем ground truth
    ax[0].imshow(image)
    ax[0].set_title("Ground Truth")
    ax[0].imshow(masks_gt[0], alpha=0.5)

    # Отображаем предсказания
    ax[1].imshow(image)
    ax[1].set_title("Prediction")
    ax[1].imshow(masks_pred[0], alpha=0.5)

    plt.show()


def custom_collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets


checkpoint_path = "sam2.1_t.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = Predictor()
predictor.args.model = checkpoint_path
predictor.setup_model(verbose=True)

criterion_mask = nn.BCEWithLogitsLoss()
criterion_box = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(predictor.model.sam_mask_decoder.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

dataset = CustomDataset(
    images_dir='dataset_coco_neuro_1/train/images',
    annotations_dir='dataset_coco_neuro_1/train/annotations'
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

num_epochs = 10
for epoch in range(num_epochs):
    predictor.model.train()
    total_loss = 0

    for images, targets in dataloader:
        optimizer.zero_grad()
        losses = []

        for img, target in zip(images, targets):
            if img.shape[:2] != (1024, 1024):
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)

            valid_indices = target["labels"] != -1
            boxes = target["boxes"][valid_indices]
            masks_gt = target["masks"][valid_indices]

            if len(boxes) == 0:
                print("no valid boxes in this image, skipping...")
                continue

            predictor.set_image(img)

            print(f"Input image shape: {img.shape}, dtype: {img.dtype}")
            print(f"Input boxes shape: {boxes.shape}, dtype: {boxes.dtype}")
            print(f"Input boxes values: {boxes}")

            masks_pred = None

            try:
                masks_pred, scores, logits = predictor.inference(
                    im=img,
                    bboxes=boxes,
                    multimask_output=False
                )
            except Exception as e:
                print(f"Ошибка в predictor.inference: {e}")

            if masks_pred is None:
                print("predictor.inference вернул None")

            try:
                masks_pred_tensor = torch.tensor(masks_pred, dtype=torch.float32, device=device)
                masks_gt_tensor = torch.tensor(masks_gt, dtype=torch.float32, device=device)
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)

                loss_mask = criterion_mask(masks_pred_tensor, masks_gt_tensor)
                loss_box = criterion_box(boxes_tensor, torch.tensor(target["boxes"][valid_indices], dtype=torch.float32, device=device))
                loss = loss_mask + loss_box
                losses.append(loss)
            except Exception as e:
                print(f"Ошибка при вычислении потерь: {e}")
                break

        if not losses:
            break

        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

torch.save(predictor.model.state_dict(), f"sam2.1_t_finetuned.pth")

sam_finetuned = build_sam(ckpt="sam2.1_t_finetuned.pth").to(device)
predictor_finetuned = Predictor()
predictor_finetuned.args.model = "sam2.1_t_finetuned.pth"
predictor_finetuned.setup_model(verbose=True)

img_test, target_test = dataset[10]
visualize_predictions(img_test, target_test, predictor_finetuned)