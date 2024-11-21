import os
import shutil
import random

images_folder = 'screen foto/dataset 2024-04-21 14_33_36/img'
labels_folder = 'screen foto/dataset 2024-04-21 14_33_36/yolo_labels'

train_images_folder = 'screen foto/dataset 2024-04-21 14_33_36/images/train'
train_labels_folder = 'screen foto/dataset 2024-04-21 14_33_36/labels/train'
val_images_folder = 'screen foto/dataset 2024-04-21 14_33_36/images/val'
val_labels_folder = 'screen foto/dataset 2024-04-21 14_33_36/labels/val'

os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

split_ratio = 0.8
label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

#Фильтруем только те файлы, у которых есть соответствующее изображение
label_files = [f for f in label_files if os.path.exists(os.path.join(images_folder, f"{os.path.splitext(f)[0]}.jpg"))]
random.shuffle(label_files)

train_size = int(len(label_files) * split_ratio)

#Разбиваем аннотации на тренировочный и валидационный наборы
train_labels = label_files[:train_size]
val_labels = label_files[train_size:]

#Копируем файлы в соответствующие папки
for label_file in train_labels:
    shutil.copy(os.path.join(labels_folder, label_file), train_labels_folder)

    img_filename = f"{os.path.splitext(label_file)[0]}.jpg"
    shutil.copy(os.path.join(images_folder, img_filename), train_images_folder)

for label_file in val_labels:
    shutil.copy(os.path.join(labels_folder, label_file), val_labels_folder)

    img_filename = f"{os.path.splitext(label_file)[0]}.jpg"
    shutil.copy(os.path.join(images_folder, img_filename), val_images_folder)

print("Разделение данных завершено.")
