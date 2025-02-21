import torch
import logging
import time
import csv
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import json
from effdet import get_efficientdet_config, create_model_from_config, DetBenchTrain
from effdet.efficientdet import HeadNet
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("training_screen.log"),
    logging.StreamHandler()
])


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
        img = Image.open(img_path).convert("RGB")
        ann_filename = img_filename.replace('.jpg', '.json')
        ann_path = os.path.join(self.annotations_dir, ann_filename)

        if not os.path.exists(ann_path):
            raise ValueError(f"Annotation file not found for image: {img_filename}")

        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        category_map = {category['id']: category['name'] for category in annotation.get('categories', [])}
        annotations = annotation.get('annotations', [])

        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if self.transforms:
            img = self.transforms(img)

        target = {
            'bbox': torch.tensor(boxes, dtype=torch.float32),
            'cls': torch.tensor(labels, dtype=torch.long),
            'img_size': (img.shape[-2], img.shape[-1])
        }

        return img, target, category_map


def custom_collate_fn(batch):
    images, targets, category_maps = zip(*batch)
    images = list(image for image in images)
    processed_targets = []

    for target in targets:
        if 'bbox' not in target or len(target['bbox']) == 0:
            print("Warning: Skipping target with missing or empty 'bbox'")
            continue
        if 'cls' not in target or len(target['cls']) == 0:
            print("Warning: Skipping target with missing or empty 'cls'")
            continue

        if target['bbox'].size(0) != target['cls'].size(0):
            raise ValueError("Number of boxes and labels must match!")

        img_size = target['img_size']
        processed_target = {
            'bbox': target['bbox'],
            'cls': target['cls'],
            'img_size': torch.tensor(img_size, dtype=torch.float32),
            'img_scale': torch.tensor([1.0], dtype=torch.float32)
        }
        processed_targets.append(processed_target)

    return images, processed_targets, category_maps


def create_model(num_classes, model_name="tf_efficientdet_d0"):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (528, 528)
    net = create_model_from_config(config, pretrained=False)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)


transform = T.Compose([
    T.ToTensor()
])

dataset = CustomDataset(
    images_dir='dataset_coco_neuro_1/train/images',
    annotations_dir='dataset_coco_neuro_1/train/annotations',
    transforms=transform
)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

img, target, category_map = dataset[0]
num_classes = len(category_map) + 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = create_model(num_classes=num_classes, model_name="tf_efficientdet_d0")
model.to(device)
print(f"Model configuration: {model.config}")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=1e-4)

with open('efficientDet_detection_result_screen.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time', 'Train/Loss', 'LR/pg0'])
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for images, targets, _ in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            images = torch.stack([image.to(device) for image in images])

            processed_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            print(f"Images shape: {images.shape}")
            if not isinstance(processed_targets, list):
                raise ValueError("Processed targets must be a list!")
            for i, t in enumerate(processed_targets):
                if not isinstance(t, dict):
                    raise ValueError(f"Target {i} is not a dictionary!")
                print(
                    f"Target {i}: bbox={t['bbox'].shape}, cls={t['cls'].shape}, img_size={t['img_size']}, img_scale={t['img_scale']}")

            optimizer.zero_grad()
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

torch.save(model.state_dict(), 'efficientDet_detection_model_screen.pth')

