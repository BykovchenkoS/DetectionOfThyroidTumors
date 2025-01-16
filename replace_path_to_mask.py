import json
import os

annotations_folder = 'dataset_coco_neuro_2/val/annotations'
new_annotations_folder = 'dataset_coco_neuro_2/val/annotations_updated'
os.makedirs(new_annotations_folder, exist_ok=True)

def update_segmentation_path(data):
    for annotation in data['annotations']:
        segmentation_path = annotation['segmentation']
        # Изменяем только путь, оставляем название файла неизменным
        new_path = segmentation_path.replace(
            "screen foto/dataset 2024-04-21 14_33_36/old_masks",
            "dataset_coco_neuro_2/masks"
        )
        annotation['segmentation'] = new_path

for filename in os.listdir(annotations_folder):
    if filename.endswith('.json'):
        filepath = os.path.join(annotations_folder, filename)

        with open(filepath, 'r') as f:
            data = json.load(f)

        update_segmentation_path(data)
        new_filepath = os.path.join(new_annotations_folder, filename)

        with open(new_filepath, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Обновленный файл сохранен: {filename}")

print("Обработка всех файлов завершена.")
