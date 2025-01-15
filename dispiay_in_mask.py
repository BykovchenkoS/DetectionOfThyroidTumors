import sys
import cv2
import numpy as np
import os
import json


def process_image(photo_number):
    json_path = f'screen foto/dataset 2024-04-21 14_33_36/shifted_json/{photo_number}.json'
    image_path = f'screen foto/dataset 2024-04-21 14_33_36/img_masks_thyroid_carotis/cropped_{photo_number}.jpg'
    masks_dir = 'screen foto/dataset 2024-04-21 14_33_36/old_masks/'

    if not os.path.exists(image_path):
        print(f"Файл изображения с номером {photo_number} не найден. Попробуйте снова.")
        return False

    if not os.path.exists(json_path):
        print(f"Файл JSON с номером {photo_number} не найден. Попробуйте снова.")
        return False

    image = cv2.imread(image_path)

    with open(json_path, 'r') as file:
        data = json.load(file)

    for obj in data['objects']:
        if obj['classTitle'] == 'Node' and obj.get("geometryType") == "polygon":
            exterior_points = np.array(obj['points']['exterior'], dtype=np.int32)
            cv2.polylines(image, [exterior_points], isClosed=True, color=(0, 255, 255), thickness=2)

        elif obj['classTitle'] == 'Node' and obj.get("geometryType") == "bitmap":
            class_name = obj.get('classTitle', 'unknown_class').replace(' ', '_')
            mask_file = f"{photo_number}_{class_name}_{obj['id']}.png"
            mask_path = os.path.join(masks_dir, mask_file)

            if not os.path.exists(mask_path):
                print(f"Маска {mask_file} для объекта ID {obj['id']} не найдена.")
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Ошибка загрузки маски {mask_file}.")
                continue

            origin = obj['bitmap'].get('origin', [0, 0])
            mask_origin = (origin[0], origin[1])

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour += np.array(mask_origin)
                cv2.drawContours(image, [contour], -1, (0, 255, 255), thickness=2)

    cv2.imshow(f"Image with Node outlined - {photo_number}", image)
    return True

while True:
    photo_number = input("Введите номер фото (или 'exit' для выхода): ").strip()

    if photo_number.lower() == 'exit':
        print("Программа завершена.")
        sys.exit()

    if not process_image(photo_number):
        continue

    key = cv2.waitKey(0)
    if key == 27:
        print("Программа завершена.")
        sys.exit()

    cv2.destroyAllWindows()
