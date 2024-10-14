import os
import json

ann_folder = 'screen foto/dataset 2024-04-21 14_33_36/clear_ann'
img_folder = 'screen foto/dataset 2024-04-21 14_33_36/img'
yolo_labels_folder = 'screen foto/dataset 2024-04-21 14_33_36/yolo_labels'

os.makedirs(yolo_labels_folder, exist_ok=True)

class_id_map = {
    13121610: 0,  #sagital
    13121611: 1,  #longitudinal
    13121612: 2,  #Thyroid tissue
    13121613: 3,  #Node
    13121614: 4,  #Carotis
    13121616: 5  #Jugular
}

for filename in os.listdir(ann_folder):
    if filename.endswith('.json'):
        json_path = os.path.join(ann_folder, filename)
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            width = data['size']['width']
            height = data['size']['height']

            yolo_label_path = os.path.join(yolo_labels_folder, f"{os.path.splitext(filename)[0]}.txt")

            with open(yolo_label_path, 'w') as yolo_file:
                for obj in data['objects']:
                    class_id = obj['classId']
                    if class_id in class_id_map:
                        class_index = class_id_map[class_id]
                        geometry_type = obj.get('geometryType', '')

                        #Обработка полигонов
                        if geometry_type == 'polygon' and 'points' in obj:
                            points = obj['points']['exterior']
                            x_min = min(point[0] for point in points)
                            x_max = max(point[0] for point in points)
                            y_min = min(point[1] for point in points)
                            y_max = max(point[1] for point in points)

                        #Обработка прямоугольников
                        elif geometry_type == 'rectangle' and 'points' in obj:
                            #Прямоугольники представлены как два угловых координаты
                            x_min = obj['points']['exterior'][0][0]
                            y_min = obj['points']['exterior'][0][1]
                            x_max = obj['points']['exterior'][1][0]
                            y_max = obj['points']['exterior'][1][1]

                        #Обработка битмапов
                        elif geometry_type == 'bitmap' and 'bitmap' in obj:
                            #Для битмапов используем их origin (верхний левый угол) и размеры
                            origin = obj['bitmap']['origin']
                            bitmap_width = obj['bitmap'].get('width', 0)
                            bitmap_height = obj['bitmap'].get('height', 0)
                            x_min = origin[0]
                            y_min = origin[1]
                            x_max = x_min + bitmap_width
                            y_max = y_min + bitmap_height

                        else:
                            print(f"Пропуск объекта с неподдерживаемым типом: {geometry_type}")
                            continue

                        x_center = (x_min + x_max) / 2 / width
                        y_center = (y_min + y_max) / 2 / height
                        bbox_width = (x_max - x_min) / width
                        bbox_height = (y_max - y_min) / height

                        yolo_file.write(f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n")
                        print(f"Записан объект для файла: {filename}")

