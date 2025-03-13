import sys
import cv2
import numpy as np
import os
import json


def process_image(photo_number):
    json_path = f'dataset_coco_neuro_2/val/annotations/{photo_number}.json'
    image_path = f'dataset_coco_neuro_2/val/images/{photo_number}.jpg'
    masks_dir = 'dataset_coco_neuro_2/masks/'

    if not os.path.exists(image_path):
        print(f"Файл изображения с номером {photo_number} не найден. Попробуйте снова.")
        return False

    if not os.path.exists(json_path):
        print(f"Файл JSON с номером {photo_number} не найден. Попробуйте снова.")
        return False

    image = cv2.imread(image_path)

    with open(json_path, 'r') as file:
        data = json.load(file)

    if 'annotations' not in data or 'categories' not in data:
        print(f"Ошибка: В файле JSON с номером {photo_number} отсутствуют ключи 'annotations' или 'categories'.")
        return False

    category_map = {cat['id']: cat['name'] for cat in data['categories']}

    mask_files = [f for f in os.listdir(masks_dir) if f.startswith(f"{photo_number}_")]

    def get_class_color(class_name):
        color_map = {
            "Carotis": (0, 165, 255),  # orange
            "Node": (0, 255, 255),  # yellow
            "Thyroid tissue": (0, 0, 255)  # red
        }
        return color_map.get(class_name, (255, 255, 255))

    polygon_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for annotation in data['annotations']:
        category_id = annotation['category_id']
        class_title = category_map.get(category_id, "Unknown")

        mask_filename = os.path.basename(annotation['segmentation'])
        mask_path = os.path.join(masks_dir, mask_filename)

        if not os.path.exists(mask_path):
            print(f"Ошибка: Маска {mask_path} не найдена.")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Ошибка: Не удалось загрузить маску {mask_path}.")
            continue

        if np.count_nonzero(mask) == 0:
            print(f"Ошибка: Маска {mask_path} пустая.")
            continue

        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"Ошибка: Контуры не найдены для маски {mask_path}.")
            continue

        color = get_class_color(class_title)
        cv2.drawContours(image, contours, -1, color, thickness=2)

    cv2.imshow("Annotated Image with Masks", image)
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