import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import torchvision.transforms as T
import pandas as pd
from coco_for_pytorch import CustomDataset
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn  # Импортируем модель

# Определяем устройство (GPU, если доступен, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализируем модель Mask R-CNN
model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=4)  # num_classes зависит от вашего набора данных
model.load_state_dict(torch.load('mask_rcnn/mask_rcnn_model.pth'))  # Загрузка весов
model.to(device)  # Переносим модель на устройство
model.eval()  # Устанавливаем модель в режим оценки

transform = T.Compose([T.ToTensor()])
dataset = CustomDataset(images_dir='dataset_coco_neuro_2/val/images',
                        annotations_dir='dataset_coco_neuro_2/val/annotations',
                        transforms=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Списки для хранения истинных меток и предсказанных значений
true_labels = []
predicted_labels = []

for images, targets, _ in dataloader:
    # Проверка типа значений в targets
    for key, value in targets.items():
        print(f"Key: {key}, Type: {type(value)}")

    images = [image.to(device) for image in images]
    targets = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in targets.items()
    }

    # Получаем предсказания модели
    with torch.no_grad():  # Отключаем градиенты для оценки
        outputs = model(images)
        # Предполагается, что outputs содержит список словарей с ключом 'labels'
        for output, target in zip(outputs, targets):
            predicted_labels.extend(output['labels'].cpu().numpy())
            true_labels.extend(target['labels'].cpu().numpy())


# Вычисляем confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2, 3])  # Пример для 4 классов

# Сохраняем confusion matrix в CSV
cm_df = pd.DataFrame(cm)
cm_df.to_csv('confusion_matrix.csv', index=False)

# Если хотите сохранить данные для каждого изображения (метки и предсказания):
data = {'True Labels': true_labels, 'Predicted Labels': predicted_labels}
df = pd.DataFrame(data)
df.to_csv('predictions_per_image.csv', index=False)
