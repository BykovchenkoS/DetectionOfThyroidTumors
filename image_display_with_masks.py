import sys
import cv2
import numpy as np
import os
import json


def process_image(photo_number):
    json_path = f'screen foto/dataset 2024-04-21 14_33_36/clear_ann/{photo_number}.json'
    image_path = f'screen foto/dataset 2024-04-21 14_33_36/img/{photo_number}.jpg'
    masks_dir = 'screen foto/dataset 2024-04-21 14_33_36/masks/'

    if not os.path.exists(image_path):
        print(f"Файл изображения с номером {photo_number} не найден. Попробуйте снова.")
        return False

    if not os.path.exists(json_path):
        print(f"Файл JSON с номером {photo_number} не найден. Попробуйте снова.")
        return False

    image = cv2.imread(image_path)

    with open(json_path, 'r') as file:
        data = json.load(file)

    mask_files = [f for f in os.listdir(masks_dir) if f.startswith(f"{photo_number}_")]

    def get_class_color(class_name):
        color_map = {
            "Carotis": (0, 165, 255),       # orange
            "Node": (0, 255, 255),          # yellow
            "Thyroid tissue": (0, 0, 255)   # red
        }
        return color_map.get(class_name, (255, 255, 255))

    polygon_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for obj in data['objects']:
        if obj.get("geometryType") == "polygon":
            exterior_points = np.array(obj['points']['exterior'], dtype=np.int32)
            cv2.fillPoly(polygon_mask, [exterior_points], 255)

    image_with_polygon = cv2.bitwise_and(image, image, mask=polygon_mask)

    for obj in data['objects']:
        if obj.get("geometryType") != "polygon":
            obj_id = obj['id']
            class_title = obj['classTitle']
            # Убираем применение смещения
            origin = [0, 0]  # всегда считаем, что origin = [0, 0]

            mask_filename = f"{photo_number}_{class_title.replace(' ', '_')}_{obj_id}.png"
            mask_path = os.path.join(masks_dir, mask_filename)

            if mask_filename in mask_files:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if mask is None or np.count_nonzero(mask) == 0:
                    print(f"Warning: Mask for {class_title} with ID {obj_id} is empty or missing.")
                else:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Без применения смещения
                    contours = [cnt for cnt in contours]  # не смещаем контуры

                    color = get_class_color(class_title)

                    cv2.drawContours(image_with_polygon, contours, -1, color, 2)

    cv2.imshow("Annotated Image with Masks and Polygon Cropped", image_with_polygon)
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
