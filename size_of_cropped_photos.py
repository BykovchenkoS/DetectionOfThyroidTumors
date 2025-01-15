import os
import json
from PIL import Image

input_dir = 'dataset_coco_neuro_2/train/annotations/'
output_dir = 'dataset_coco_neuro_2/train/annotations_/'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r') as f:
            data = json.load(f)

        for image_info in data['images']:
            image_path = image_info['file_name']

            with Image.open(image_path) as img:
                width, height = img.size

            image_info['width'] = width
            image_info['height'] = height

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Файл {filename} обновлен и сохранен в '{output_dir}'.")
