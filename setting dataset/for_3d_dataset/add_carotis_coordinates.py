import cv2
import numpy as np
import json
import os
import shutil

input_annotation_dir_1 = "../../dataset_coco_neuro_1/ann_neuro_1"
input_annotation_dir_2 = "../../dataset_coco_neuro_2/ann_neuro_2"

output_image_dir = "../../dataset_node/img"
output_annotation_dir = "../../dataset_node/ann"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

annotation_files_1 = {f for f in os.listdir(input_annotation_dir_1) if f.endswith(".json")}
annotation_files_2 = {f for f in os.listdir(input_annotation_dir_2) if f.endswith(".json")}

common_annotation_files = annotation_files_1.intersection(annotation_files_2)

if not common_annotation_files:
    print("Нет общих аннотаций в dataset_coco_neuro_1/ann_neuro_1 и dataset_coco_neuro_2/ann_neuro_2.")
    exit()

for annotation_file in common_annotation_files:
    annotation_path_1 = os.path.join(input_annotation_dir_1, annotation_file)
    annotation_path_2 = os.path.join(input_annotation_dir_2, annotation_file)

    with open(annotation_path_1, "r") as f:
        annotation_data_1 = json.load(f)

    bbox_thyroid = None
    bbox_carotis = None
    carotis_found = False

    for ann in annotation_data_1["annotations"]:
        if ann["category_id"] == 1:  # Thyroid tissue
            bbox_thyroid = ann["bbox"]
        elif ann["category_id"] == 2:  # Carotis
            bbox_carotis = ann["bbox"]
            carotis_found = True

    if bbox_thyroid is None:
        print(f"Bounding box для Thyroid tissue не найден в файле {annotation_file}. Пропуск файла.")
        continue

    if carotis_found:
        image_path = annotation_data_1["images"][0]["file_name"]
        image = cv2.imread(image_path)

        # Объединение bounding boxes
        x_min = int(min(bbox_thyroid[0], bbox_carotis[0]))
        y_min = int(min(bbox_thyroid[1], bbox_carotis[1]))
        x_max = int(max(bbox_thyroid[0] + bbox_thyroid[2], bbox_carotis[0] + bbox_carotis[2]))
        y_max = int(max(bbox_thyroid[1] + bbox_thyroid[3], bbox_carotis[1] + bbox_carotis[3]))

        # Обрезка изображения
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Растягивание изображения по ширине до 528 пикселей
        target_width = 528
        height, width = cropped_image.shape[:2]
        scale_factor = target_width / width
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(cropped_image, (target_width, new_height))

        # Добавление паддингов сверху и снизу
        target_size = 528
        top_pad = (target_size - new_height) // 2
        bottom_pad = target_size - new_height - top_pad

        padded_image = cv2.copyMakeBorder(
            resized_image,
            top=top_pad,
            bottom=bottom_pad,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Расчет новых координат для Carotis
        original_x_min = bbox_carotis[0]
        original_y_min = bbox_carotis[1]
        original_width = bbox_carotis[2]
        original_height = bbox_carotis[3]

        # Смещение координат из-за обрезки
        cropped_x_min = original_x_min - x_min
        cropped_y_min = original_y_min - y_min
        cropped_x_max = cropped_x_min + original_width
        cropped_y_max = cropped_y_min + original_height

        # Масштабирование координат из-за растягивания
        scaled_x_min = int(cropped_x_min * scale_factor)
        scaled_y_min = int(cropped_y_min * scale_factor)
        scaled_x_max = int(cropped_x_max * scale_factor)
        scaled_y_max = int(cropped_y_max * scale_factor)

        # Смещение координат из-за добавления паддингов
        final_x_min = scaled_x_min
        final_y_min = scaled_y_min + top_pad
        final_x_max = scaled_x_max
        final_y_max = scaled_y_max + top_pad

        # Новые координаты bounding box для Carotis
        new_bbox_carotis = [final_x_min, final_y_min, final_x_max - final_x_min, final_y_max - final_y_min]

        output_image_path = os.path.join(output_image_dir, annotation_file.replace(".json", ".jpg"))
        cv2.imwrite(output_image_path, padded_image)

        print(f"Обработанное изображение сохранено в {output_image_path}")

        with open(annotation_path_2, "r") as f:
            annotation_data_2 = json.load(f)

        # Добавление новой категории "Carotis"
        new_category_id = max([cat["id"] for cat in annotation_data_2["categories"]]) + 1
        new_category = {
            "id": new_category_id,
            "name": "Carotis",
            "supercategory": "organ"
        }
        annotation_data_2["categories"].append(new_category)

        new_annotation_id = max([ann["id"] for ann in annotation_data_2["annotations"]]) + 1
        new_annotation = {
            "id": new_annotation_id,
            "image_id": int(annotation_file.split(".")[0]),
            "category_id": new_category_id,
            "segmentation": [],
            "area": new_bbox_carotis[2] * new_bbox_carotis[3],
            "bbox": new_bbox_carotis,
            "iscrowd": 0
        }
        annotation_data_2["annotations"].append(new_annotation)

        output_annotation_path = os.path.join(output_annotation_dir, annotation_file)
        with open(output_annotation_path, "w") as f:
            json.dump(annotation_data_2, f, indent=4)

        print(f"Новая категория 'Carotis' и аннотация успешно сохранены в {output_annotation_path}")

    else:
        output_annotation_path = os.path.join(output_annotation_dir, annotation_file)
        shutil.copy(annotation_path_2, output_annotation_path)

        # Сохранение обрезанного изображения только по Thyroid tissue
        image_path = annotation_data_1["images"][0]["file_name"]
        image = cv2.imread(image_path)

        # Обрезка изображения по Thyroid tissue
        x_min = int(bbox_thyroid[0])
        y_min = int(bbox_thyroid[1])
        x_max = int(bbox_thyroid[0] + bbox_thyroid[2])
        y_max = int(bbox_thyroid[1] + bbox_thyroid[3])

        cropped_image = image[y_min:y_max, x_min:x_max]

        # Растягивание изображения по ширине до 528 пикселей
        target_width = 528
        height, width = cropped_image.shape[:2]
        scale_factor = target_width / width
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(cropped_image, (target_width, new_height))

        # Добавление паддингов сверху и снизу
        target_size = 528
        top_pad = (target_size - new_height) // 2
        bottom_pad = target_size - new_height - top_pad

        padded_image = cv2.copyMakeBorder(
            resized_image,
            top=top_pad,
            bottom=bottom_pad,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        output_image_path = os.path.join(output_image_dir, annotation_file.replace(".json", ".jpg"))
        cv2.imwrite(output_image_path, padded_image)

        print(f"Обрезанное изображение сохранено в {output_image_path}")
        print(f"Аннотация скопирована без изменений в {output_annotation_path}")


# ПОСЛЕ ЗАПУСТИ fix_id_size.py
