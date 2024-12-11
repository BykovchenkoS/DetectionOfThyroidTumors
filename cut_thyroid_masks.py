import os
import cv2
import numpy as np
import json
import re

images_path = "screen foto/dataset 2024-04-21 14_33_36/img"
masks_path = "screen foto/dataset 2024-04-21 14_33_36/masks"
json_path = "screen foto/dataset 2024-04-21 14_33_36/clear_ann"
output_path = "screen foto/dataset 2024-04-21 14_33_36/img_masks_thyroid_carotis"

os.makedirs(output_path, exist_ok=True)

image_files = [i for i in os.listdir(images_path) if i.endswith('.jpg')]
mask_files = os.listdir(masks_path)

processed_count = 0

for image_file in image_files:
    image_name, _ = os.path.splitext(image_file)
    json_file = f"{image_name}.json"
    image_path = os.path.join(images_path, image_file)
    json_file_path = os.path.join(json_path, json_file)

    image = cv2.imread(image_path)

    try:
        with open(json_file_path, 'r') as json_f:
            json_data = json.load(json_f)
            regions = []

            for obj in json_data.get("objects", []):
                class_title = obj.get("classTitle")
                if class_title in ["Thyroid tissue", "Carotis"] and "bitmap" in obj:
                    origin = obj["bitmap"].get("origin", [0, 0])

                    mask_pattern = re.compile(f"{re.escape(image_name)}_{class_title.replace(' ', '_')}.*\.png")
                    matching_masks = [mask for mask in mask_files if mask_pattern.match(mask)]

                    if matching_masks:
                        mask_path = os.path.join(masks_path, matching_masks[0])
                        regions.append({"origin": origin, "mask_path": mask_path})
                        print(f"Координаты origin для {image_file} ({class_title}): {origin}")
                    else:
                        print(f"Маска для {class_title} не найдена для файла {image_file}.")

            if not regions:
                print(f"Не найдено 'origin' или маски для {image_file} в классах 'Thyroid tissue' и 'Carotis'.")
                continue
    except Exception as e:
        print(f"Ошибка при чтении JSON для {image_file}: {e}")
        continue

    x_min_global, y_min_global = float("inf"), float("inf")
    x_max_global, y_max_global = float("-inf"), float("-inf")

    for region in regions:
        mask = cv2.imread(region["mask_path"], cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        y_indices, x_indices = np.where(binary_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            print(f"Область маски не найдена в {region['mask_path']}.")
            continue

        origin = region["origin"]
        x_min = x_indices.min() + origin[0]
        x_max = x_indices.max() + origin[0]
        y_min = y_indices.min() + origin[1]
        y_max = y_indices.max() + origin[1]

        x_min_global = min(x_min_global, x_min)
        y_min_global = min(y_min_global, y_min)
        x_max_global = max(x_max_global, x_max)
        y_max_global = max(y_max_global, y_max)

    if x_min_global == float("inf") or y_min_global == float("inf"):
        print(f"Не удалось вычислить bounding box для {image_file}.")
        continue

    cropped_image = image[y_min_global:y_max_global + 1, x_min_global:x_max_global + 1]

    output_file = os.path.join(output_path, f"cropped_{image_file}")
    cv2.imwrite(output_file, cropped_image)
    processed_count += 1
    print(f"Сохраненное обрезанное изображение: {output_file}")

print(f"Обработано изображений: {processed_count} из {len(image_files)}")
