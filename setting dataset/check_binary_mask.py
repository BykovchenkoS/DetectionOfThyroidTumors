import cv2
import numpy as np
import os

input_dir = "../dataset_for_search_1/old_masks"
output_dir = "../dataset_for_search_1/masks"

os.makedirs(output_dir, exist_ok=True)

mask_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

for mask_file in mask_files:
    input_path = os.path.join(input_dir, mask_file)

    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Ошибка загрузки файла: {input_path}")
        continue

    print(f"Обработка маски: {mask_file}")
    print(f"Размер маски: {mask.shape}")  # размер маски (height, width)
    print(f"Тип данных: {mask.dtype}")  # тип данных (должно быть uint8)
    print(f"Пиксели: {np.unique(mask)}")  # уникальные значения пикселей в маске

    # бинаризация маски
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    print(f"Пиксели после бинаризации: {np.unique(binary_mask)}")  # должно быть [0, 255]

    output_path = os.path.join(output_dir, mask_file)
    cv2.imwrite(output_path, binary_mask)
    print(f"Исправленная маска сохранена по пути: {output_path}\n")

print("Обработка завершена")

