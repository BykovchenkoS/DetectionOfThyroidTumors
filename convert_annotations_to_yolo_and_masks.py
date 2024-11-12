import os
import json
import numpy as np
import cv2

ann_folder = 'screen foto/dataset 2024-04-21 14_33_36/clear_ann'
img_folder = 'screen foto/dataset 2024-04-21 14_33_36/img'
yolo_labels_folder = 'screen foto/dataset 2024-04-21 14_33_36/yolo_labels'
masks_folder = 'screen foto/dataset 2024-04-21 14_33_36/masks'
os.makedirs(yolo_labels_folder, exist_ok=True)
os.makedirs(masks_folder, exist_ok=True)

class_id_map = {
    13121610: 0,  # sagital
    13121611: 0,  # longitudinal
    13121612: 1,  # Thyroid tissue
    13121613: 2,  # Node
    13121614: 3,  # Carotis
    13121616: 4  # Jugular
}

# Создаем маску для выделения только объектов, относящихся к щитовидной железе
thyroid_class_ids = {13121612}

for filename in os.listdir(ann_folder):
    if filename.endswith('.json'):
        json_path = os.path.join(ann_folder, filename)
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            width = data['size']['width']
            height = data['size']['height']

            yolo_label_path = os.path.join(yolo_labels_folder, f"{os.path.splitext(filename)[0]}.txt")

            # Создаем маску (бинарное изображение)
            mask = np.zeros((height, width), dtype=np.uint8)

            with open(yolo_label_path, 'w') as yolo_file:
                for obj in data['objects']:
                    class_id = obj['classId']
                    geometry_type = obj.get('geometryType', '')

                    # Объединяем sagital и longitudinal в один класс
                    if class_id in {13121610, 13121611}:
                        class_id = 13121610

                    # Если объект относится к щитовидной железе, обрабатываем его
                    if class_id in class_id_map:
                        class_index = class_id_map[class_id]

                        if geometry_type == 'polygon' and 'points' in obj:
                            points = obj['points']['exterior']
                            x_min = min(point[0] for point in points)
                            x_max = max(point[0] for point in points)
                            y_min = min(point[1] for point in points)
                            y_max = max(point[1] for point in points)
                            points = np.array(points, dtype=np.int32)
                            cv2.fillPoly(mask, [points], 255)

                        elif geometry_type == 'rectangle' and 'points' in obj:
                            x_min = obj['points']['exterior'][0][0]
                            y_min = obj['points']['exterior'][0][1]
                            x_max = obj['points']['exterior'][1][0]
                            y_max = obj['points']['exterior'][1][1]
                            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=cv2.FILLED)

                        elif geometry_type == 'bitmap' and 'bitmap' in obj:
                            origin = obj['bitmap']['origin']
                            bitmap_width = obj['bitmap'].get('width', 0)
                            bitmap_height = obj['bitmap'].get('height', 0)
                            x_min = origin[0]
                            y_min = origin[1]
                            x_max = x_min + bitmap_width
                            y_max = y_min + bitmap_height
                            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=cv2.FILLED)

                        else:
                            print(f"Пропуск объекта с неподдерживаемым типом: {geometry_type}")
                            continue

                        x_center = (x_min + x_max) / 2 / width
                        y_center = (y_min + y_max) / 2 / height
                        bbox_width = (x_max - x_min) / width
                        bbox_height = (y_max - y_min) / height

                        # Записываем метки YOLO для 1-ой модели (Thyroid tissue)
                        if class_id == 13121612:
                            yolo_file.write(f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n")

                        # Записываем метки YOLO для 2-ой модели (Node, Carotis, Jugular)
                        elif class_id in {13121613, 13121614, 13121616}:
                            yolo_file.write(f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n")

                mask_path = os.path.join(masks_folder, f"{os.path.splitext(filename)[0]}_mask.png")
                cv2.imwrite(mask_path, mask)

                print(f"Записан объект для файла: {filename} и маска сохранена.")
