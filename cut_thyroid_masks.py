import os
import cv2
import numpy as np
import json

images_path = "screen foto/dataset 2024-04-21 14_33_36/img"
masks_path = "screen foto/dataset 2024-04-21 14_33_36/masks"
json_path = "screen foto/dataset 2024-04-21 14_33_36/clear_ann"
output_path = "screen foto/dataset 2024-04-21 14_33_36/img_masks_thyroid"

os.makedirs(output_path, exist_ok=True)

image_files = [i for i in os.listdir(images_path) if i.endswith('.jpg')]
mask_files = [i for i in os.listdir(masks_path) if 'Thyroid_tissue' in i]

processed_count = 0

for image_file in image_files:
    image_name, _ = os.path.splitext(image_file)
    matching_masks = [mask for mask in mask_files if mask.startswith(image_name + '_Thyroid_tissue')]

    if not matching_masks:
        print(f"Не найдено подходящей маски для {image_file}.")
        continue

    mask_file = matching_masks[0]
    json_file = f"{image_name}.json"

    image_path = os.path.join(images_path, image_file)
    mask_path = os.path.join(masks_path, mask_file)
    json_file_path = os.path.join(json_path, json_file)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    try:
        with open(json_file_path, 'r') as json_f:
            json_data = json.load(json_f)
            origin = None
            for obj in json_data.get("objects", []):
                if obj.get("classTitle") == "Thyroid tissue" and "bitmap" in obj:
                    origin = obj["bitmap"].get("origin", [0, 0])  # Извлекаем origin из bitmap
                    break
            if origin:
                print(f"Координаты origin для {image_file}: {origin}")
            else:
                print(f"Не найдено 'origin' для {image_file} в классе 'Thyroid tissue'")
    except Exception as e:
        print(f"Ошибка при чтении JSON для {image_file}: {e}")
        continue

    if not origin:
        continue

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    y_indices, x_indices = np.where(binary_mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        print(f"Область маски не найдена в {mask_file}.")
        continue

    x_min, x_max = x_indices.min() + origin[0], x_indices.max() + origin[0]
    y_min, y_max = y_indices.min() + origin[1], y_indices.max() + origin[1]

    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

    output_file = os.path.join(output_path, f"cropped_{image_file}")
    cv2.imwrite(output_file, cropped_image)
    processed_count += 1
    print(f"Сохраненное обрезанное изображение: {output_file}")

print(f"Обработано изображений: {processed_count} из {len(image_files)}")
