import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def analyze_inclusions(image, mask):
    # Извлекаем область узла
    node_region = cv2.bitwise_and(image, image, mask=mask)

    # Применяем пороговую обработку для выделения включений
    _, thresholded = cv2.threshold(node_region, 200, 255, cv2.THRESH_BINARY)

    # Находим контуры включений
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Контуры не найдены.")
        return ["Нет"]

    inclusion_types = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(f"Контур {i + 1}: Площадь = {area}")

        if area < 10:  # Игнорируем слишком маленькие контуры
            print(f"Контур {i + 1} пропущен (площадь < 10).")
            continue

        # Вычисляем компактность контура
        perimeter = cv2.arcLength(contour, True)
        compactness = (perimeter ** 2) / (4 * np.pi * area)
        print(f"Контур {i + 1}: Компактность = {compactness}")

        # Вычисляем среднюю интенсивность включения
        mask_inclusion = np.zeros_like(image)
        cv2.drawContours(mask_inclusion, [contour], -1, 255, thickness=cv2.FILLED)
        inclusion_pixels = image[mask_inclusion > 0]
        mean_intensity = np.mean(inclusion_pixels)
        print(f"Контур {i + 1}: Средняя интенсивность = {mean_intensity}")

        if area > 1000 and compactness < 1.5:
            # Проверяем, есть ли "хвост кометы" (V-образная форма)
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None and len(defects) > 2:  # V-образная форма
                inclusion_types.append("Крупный артефакт с 'хвостом кометы'")
                print(f"Контур {i + 1}: Крупный артефакт с 'хвостом кометы'")
            else:
                inclusion_types.append("Макрокальцификации")
                print(f"Контур {i + 1}: Макрокальцификации")
        elif area > 100 and compactness < 1.8:
            # Проверяем, находится ли включение на краю узла
            edge_distance = cv2.pointPolygonTest(contour, (image.shape[1] // 2, image.shape[0] // 2), True)
            if edge_distance < 0:  # Включение на краю
                inclusion_types.append("Периферическая кальцификация")
                print(f"Контур {i + 1}: Периферическая кальцификация")
            else:
                inclusion_types.append("Точечные эхогенные очаги")
                print(f"Контур {i + 1}: Точечные эхогенные очаги")
        elif area > 50 and mean_intensity > 200:
            inclusion_types.append("Точечные эхогенные очаги")
            print(f"Контур {i + 1}: Точечные эхогенные очаги")
        else:
            print(f"Контур {i + 1}: Не удалось классифицировать.")

    unique_inclusions = list(set(inclusion_types)) if inclusion_types else ["Нет"]
    print(f"Итоговые типы включений: {unique_inclusions}")
    return unique_inclusions


def process_image_by_number(image_number):
    cropped_images_dir = '../dataset_coco_neuro_3/images_neuro_3'
    cropped_masks_dir = '../dataset_coco_neuro_3/masks'
    full_images_dir = '../dataset_coco_neuro_1/images_neuro_1'
    full_masks_dir = '../dataset_coco_neuro_1/masks'

    cropped_image_path = os.path.join(cropped_images_dir, f"{image_number}.jpg")
    cropped_mask_files = [f for f in os.listdir(cropped_masks_dir) if f.startswith(f"{image_number}_Node")]

    if not cropped_mask_files:
        print(f"Маска узла для изображения {image_number} не найдена.")
        return

    cropped_mask_path = os.path.join(cropped_masks_dir, cropped_mask_files[0])

    full_image_path = os.path.join(full_images_dir, f"{image_number}.jpg")
    full_mask_thyroid_files = [f for f in os.listdir(full_masks_dir) if f.startswith(f"{image_number}_Thyroid")]
    full_mask_carotis_files = [f for f in os.listdir(full_masks_dir) if f.startswith(f"{image_number}_Carotis")]

    if not full_mask_thyroid_files:
        print(f"Маска thyroid для изображения {image_number} не найдена.")
        return

    full_mask_thyroid_path = os.path.join(full_masks_dir, full_mask_thyroid_files[0])

    full_mask_carotis_path = None
    if full_mask_carotis_files:
        full_mask_carotis_path = os.path.join(full_masks_dir, full_mask_carotis_files[0])

    if not os.path.exists(cropped_image_path):
        print(f"Обрезанное изображение {cropped_image_path} не найдено.")
        return
    if not os.path.exists(cropped_mask_path):
        print(f"Маска узла {cropped_mask_path} не найдена.")
        return
    if not os.path.exists(full_image_path):
        print(f"Полное изображение {full_image_path} не найдено.")
        return
    if not os.path.exists(full_mask_thyroid_path):
        print(f"Маска thyroid {full_mask_thyroid_path} не найдена.")
        return

    cropped_image = cv2.imread(cropped_image_path, cv2.IMREAD_GRAYSCALE)
    cropped_mask = cv2.imread(cropped_mask_path, cv2.IMREAD_GRAYSCALE)

    full_image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
    full_mask_thyroid = cv2.imread(full_mask_thyroid_path, cv2.IMREAD_GRAYSCALE)
    full_mask_carotis = None
    if full_mask_carotis_path and os.path.exists(full_mask_carotis_path):
        full_mask_carotis = cv2.imread(full_mask_carotis_path, cv2.IMREAD_GRAYSCALE)

    inclusion_type = analyze_inclusions(cropped_image, cropped_mask)

    node_in_thyroid = np.any(cv2.bitwise_and(cropped_mask, full_mask_thyroid))
    node_near_carotis = False
    if full_mask_carotis is not None:
        node_near_carotis = np.any(cv2.bitwise_and(cropped_mask, full_mask_carotis))

    print(f"Изображение {image_number}:")
    print(f"  Тип включения: {inclusion_type}")
    print(f"  Узел находится внутри thyroid: {'Да' if node_in_thyroid else 'Нет'}")
    print(f"  Узел находится рядом с carotis: {'Да' if node_near_carotis else 'Нет (или маска carotis отсутствует)'}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(full_image, cmap='gray')

    plt.contour(full_mask_thyroid, colors='green', levels=[0.5])
    if full_mask_carotis is not None:
        plt.contour(full_mask_carotis, colors='blue', levels=[0.5])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Thyroid'),
        Line2D([0], [0], color='blue', lw=2, label='Carotis') if full_mask_carotis is not None else None
    ]

    legend_elements = [elem for elem in legend_elements if elem is not None]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(f"Исходное изображение {image_number}: Thyroid, Carotis")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_image, cmap='gray')
    plt.contour(cropped_mask, colors='red', levels=[0.5])

    plt.legend(handles=[Line2D([0], [0], color='red', lw=2, label='Node')], loc='upper right')
    plt.title(f"Обрезанное изображение {image_number}: Узел")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


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