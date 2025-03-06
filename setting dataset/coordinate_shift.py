import json
import os
import cv2
import numpy as np
import re

def shift_node_coordinates(json_data, x_min, y_min):
    for obj in json_data.get("objects", []):
        if obj.get("classTitle") == "Node" and "bitmap" in obj:
            origin = obj["bitmap"].get("origin", [0, 0])
            obj["bitmap"]["origin"] = [origin[0] - x_min, origin[1] - y_min]
            if "points" in obj and "exterior" in obj["points"]:
                for i, point in enumerate(obj["points"]["exterior"]):
                    obj["points"]["exterior"][i] = [point[0] - x_min, point[1] - y_min]
    return json_data

def convert_int64_to_int(obj):
    if isinstance(obj, dict):
        return {key: convert_int64_to_int(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

images_path = "../screen foto/dataset 2024-04-21 14_33_36/img"
json_path = "../screen foto/dataset 2024-04-21 14_33_36/clear_ann"
output_json_path = "../screen foto/dataset 2024-04-21 14_33_36/shifted_json"

os.makedirs(output_json_path, exist_ok=True)

image_files = [i for i in os.listdir(images_path) if i.endswith('.jpg')]

for image_file in image_files:
    image_name, _ = os.path.splitext(image_file)
    json_file = f"{image_name}.json"
    json_file_path = os.path.join(json_path, json_file)

    try:
        with open(json_file_path, 'r') as json_f:
            json_data = json.load(json_f)
            regions = []
            for obj in json_data.get("objects", []):
                class_title = obj.get("classTitle")
                if class_title in ["Thyroid tissue", "Carotis"] and "bitmap" in obj:
                    origin = obj["bitmap"].get("origin", [0, 0])
                    mask_pattern = f"{re.escape(image_name)}_{re.escape(class_title.replace(' ', '_'))}.*\.png"
                    matching_masks = [mask for mask in os.listdir("../screen foto/dataset 2024-04-21 14_33_36/masks") if re.match(mask_pattern, mask)]

                    if matching_masks:
                        mask_path = os.path.join("../screen foto/dataset 2024-04-21 14_33_36/masks", matching_masks[0])
                        regions.append({"origin": origin, "mask_path": mask_path})

            if not regions:
                print(f"Не найдено 'origin' или маски для {image_file}.")
                continue

            x_min_global, y_min_global = float("inf"), float("inf")
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

            if x_min_global == float("inf") or y_min_global == float("inf"):
                print(f"Не удалось вычислить bounding box для {image_file}.")
                continue

            updated_json_data = shift_node_coordinates(json_data, x_min_global, y_min_global)

            updated_json_data = convert_int64_to_int(updated_json_data)

            output_json_file = os.path.join(output_json_path, json_file)
            with open(output_json_file, 'w') as out_f:
                json.dump(updated_json_data, out_f, indent=4)

            print(f"Обновленный JSON сохранен: {output_json_file}")

    except Exception as e:
        print(f"Ошибка при обработке JSON для {image_file}: {e}")
