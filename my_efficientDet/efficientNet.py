import torch
import torchvision
import logging
import time
import csv
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from torchvision import transforms as T

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
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        # Получаем category_map
        category_map = {category['id']: category['name'] for category in annotation['categories']}

        # Берем первую аннотацию (если есть несколько, берем первую)
        label = annotation['annotations'][0]['category_id'] if annotation['annotations'] else 0

        if self.transforms:
            img = self.transforms(img)

        return img, label, category_map


def create_model(num_classes):
    model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    return model


transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(
    images_dir='dataset_coco_neuro_1/train/images',
    annotations_dir='dataset_coco_neuro_1/train/annotations',
    transforms=transform
)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

img, label, category_map = dataset[0]
num_classes = len(category_map) + 1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = create_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
criterion = torch.nn.CrossEntropyLoss()

with open('efficientNet_classification_result_screen.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time', 'Train/Loss', 'LR/pg0'])

    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for images, labels, _ in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        writer.writerow([epoch + 1, epoch_time, epoch_loss / len(data_loader), optimizer.param_groups[0]['lr']])
        file.flush()

        log_message = (f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(data_loader):.4f}")
        logging.info(log_message)
        print(log_message)

# Сохранение модели
torch.save(model.state_dict(), 'efficientNet_classification_model_screen.pth')