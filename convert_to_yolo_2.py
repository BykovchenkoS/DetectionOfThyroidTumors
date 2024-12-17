import cv2
import numpy as np
import os
import json

def normalize_coordinates(x, y, img_width, img_height):
    return x / img_width, y / img_height

def save_yolo_annotation(photo_number, annotations, output_dir):
    yolo_file_path = os.path.join(output_dir, f"{photo_number}.txt")
    with open(yolo_file_path, 'w') as file:
        for class_id, bbox in annotations:
            file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    print(f"YOLO разметка сохранена: {yolo_file_path}")

def process_image(json_path, image_path, masks_dir, yolo_output_dir):
    photo_number = os.path.splitext(os.path.basename(json_path))[0]

    if not os.path.exists(image_path):
        print(f"Изображение {photo_number} не найдено, пропускаем.")
        return

    with open(json_path, 'r') as file:
        data = json.load(file)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки изображения {photo_number}. Пропускаем.")
        return

    img_height, img_width = image.shape[:2]
    annotations = []

    for obj in data['objects']:
        if obj['classTitle'] == 'Node':
            if obj.get("geometryType") == "polygon":
                exterior_points = np.array(obj['points']['exterior'], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(exterior_points)
            elif obj.get("geometryType") == "bitmap":
                mask_file = f"{photo_number}_Node_{obj['id']}.png"
                mask_path = os.path.join(masks_dir, mask_file)

                if not os.path.exists(mask_path):
                    print(f"Маска {mask_file} не найдена.")
                    continue

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Ошибка загрузки маски {mask_file}.")
                    continue

                origin = obj['bitmap'].get('origin', [0, 0])
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contours[0])
                x += origin[0]
                y += origin[1]

            x_center, y_center = normalize_coordinates(x+w / 2, y+h / 2, img_width, img_height)
            norm_width, norm_height = normalize_coordinates(w, h, img_width, img_height)

            annotations.append((0, (x_center, y_center, norm_width, norm_height)))

    if annotations:
        save_yolo_annotation(photo_number, annotations, yolo_output_dir)

def process_all_json():
    json_dir = 'screen foto/dataset 2024-04-21 14_33_36/shifted_json/'
    image_dir = 'screen foto/dataset 2024-04-21 14_33_36/img_masks_thyroid_carotis/'
    masks_dir = 'screen foto/dataset 2024-04-21 14_33_36/masks/'
    yolo_output_dir = 'screen foto/dataset 2024-04-21 14_33_36/yolo_annotations/'

    os.makedirs(yolo_output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print("JSON-файлы не найдены!")
        return

    print(f"Найдено {len(json_files)} JSON-файлов. Начинаем обработку...")
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        image_path = os.path.join(image_dir, f"cropped_{os.path.splitext(json_file)[0]}.jpg")
        process_image(json_path, image_path, masks_dir, yolo_output_dir)

    print("Обработка завершена. YOLO разметка сохранена в папке 'yolo_annotations/'.")

if __name__ == "__main__":
    process_all_json()
