import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def determine_echogenicity(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Вычисление средней интенсивности пикселей в области узла
    mean_intensity = np.mean(masked_image[mask > 0])

    if mean_intensity > 180:
        return "Гиперэхогенный"
    elif mean_intensity > 120:
        return "Изоэхогенный"
    elif mean_intensity > 60:
        return "Гипоэхогенный"
    else:
        return "Анэхогенный"


def process_image_by_number(image_number):
    images_dir = 'dataset_coco_neuro_3/images_neuro_3'
    masks_dir = 'dataset_coco_neuro_3/masks'

    image_path = os.path.join(images_dir, f"{image_number}.jpg")
    mask_filename = [f for f in os.listdir(masks_dir) if f.startswith(f"{image_number}_")]

    if not mask_filename:
        print(f"Маска для изображения {image_number}.jpg не найдена.")
        return

    mask_path = os.path.join(masks_dir, mask_filename[0])

    if not os.path.exists(image_path):
        print(f"Изображение {image_path} не найдено.")
        return
    if not os.path.exists(mask_path):
        print(f"Маска {mask_path} не найдена.")
        return

    echogenicity = determine_echogenicity(image_path, mask_path)

    thyroid_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask_node = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(thyroid_image, cv2.COLOR_GRAY2RGB))
    plt.contour(mask_node, colors='red', levels=[0.5])

    plt.text(10, 30, f"Эхогенность узла: {echogenicity}", color='white', backgroundcolor='black', fontsize=12)
    plt.title(f"Изображение {image_number}: Эхогенность узла - {echogenicity}")
    plt.show()

    print(f"Изображение {image_number}: Эхогенность узла - {echogenicity}")


while True:
    user_input = input("Введите номер фото (или 'exit' для выхода): ")

    if user_input.lower() == 'exit':
        print("Программа завершена.")
        break

    if not user_input.isdigit():
        print("Пожалуйста, введите корректный номер фото.")
        continue

    image_number = int(user_input)
    process_image_by_number(image_number)
