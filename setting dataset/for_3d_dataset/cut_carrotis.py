from PIL import Image
import numpy as np
import json
import shutil
import os

annotations_dir = "../../dataset_node/ann"
output_images_dir = "dataset_coco_neuro_3/img"
output_masks_dir = "../../dataset_coco_neuro_3/masks"
output_annotations_dir = "../../dataset_coco_neuro_3/ann"


os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)
os.makedirs(output_annotations_dir, exist_ok=True)

# Счетчик пропущенных файлов из-за пересечения
skipped_files_count = 0

for annotation_file in os.listdir(annotations_dir):
    if not annotation_file.endswith(".json"):
        continue

    annotation_path = os.path.join(annotations_dir, annotation_file)
    file_id = os.path.splitext(annotation_file)[0]

    print(f"Обработка файла: {annotation_file}")

    with open(annotation_path, "r") as f:
        data = json.load(f)

    carotis_found = any(annotation["category_id"] == 2 for annotation in data["annotations"])

    if not carotis_found:
        print(f"Класс Carotis не найден в аннотации {annotation_file}. Копирую исходные файлы.")

        image_path = data["images"][0]["file_name"]
        mask_path = None

        for annotation in data["annotations"]:
            if annotation["category_id"] == 1:
                mask_path = annotation["segmentation"]
                break

        output_image_path = os.path.join(output_images_dir, os.path.basename(image_path))
        output_mask_path = os.path.join(output_masks_dir, os.path.basename(mask_path)) if mask_path else None
        output_annotation_path = os.path.join(output_annotations_dir, annotation_file)

        shutil.copy(image_path, output_image_path)
        print(f"Изображение скопировано: {output_image_path}")

        if mask_path:
            shutil.copy(mask_path, output_mask_path)
            print(f"Маска скопирована: {output_mask_path}")
        else:
            print("Маска для Node не найдена. Пропускаю копирование маски.")

        with open(output_annotation_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Аннотация сохранена: {output_annotation_path}")

    else:
        carotis_bbox = None
        node_bbox = None
        node_mask_path = None

        for annotation in data["annotations"]:
            if annotation["category_id"] == 2:
                carotis_bbox = annotation["bbox"]
            elif annotation["category_id"] == 1:
                node_bbox = annotation["bbox"]
                node_mask_path = annotation["segmentation"]

        if not carotis_bbox or not node_bbox or not node_mask_path:
            raise ValueError(f"Не найдены аннотации или маска для Node в файле {annotation_file}")

        # Проверка на пересечение bounding box'ов
        if (
            carotis_bbox[0] < node_bbox[0] + node_bbox[2]
            and carotis_bbox[0] + carotis_bbox[2] > node_bbox[0]
            and carotis_bbox[1] < node_bbox[1] + node_bbox[3]
            and carotis_bbox[1] + carotis_bbox[3] > node_bbox[1]
        ):
            print(f"Carotis и Node пересекаются в файле {annotation_file}. Пропускаю обработку.")
            skipped_files_count += 1
            continue

        image_path = data["images"][0]["file_name"]
        image = Image.open(image_path)

        image_width, image_height = image.size

        if node_bbox[0] + node_bbox[2] < carotis_bbox[0]:  # Node слева от Carotis
            print("Node находится слева от Carotis.")
            crop_left = 0
            crop_right = int(carotis_bbox[0])  # Левая граница Carotis
        elif carotis_bbox[0] + carotis_bbox[2] < node_bbox[0]:  # Node справа от Carotis
            print("Node находится справа от Carotis.")
            crop_left = int(carotis_bbox[0] + carotis_bbox[2])  # Правая граница Carotis
            crop_right = image_width  # Ширина изображения
        else:  # Проверка на вертикальное расположение
            if node_bbox[1] + node_bbox[3] < carotis_bbox[1]:  # Node выше Carotis
                print("Node находится выше Carotis.")
                crop_top = 0
                crop_bottom = int(carotis_bbox[1])  # Верхняя граница Carotis
            elif carotis_bbox[1] + carotis_bbox[3] < node_bbox[1]:  # Node ниже Carotis
                print("Node находится ниже Carotis.")
                crop_top = int(carotis_bbox[1] + carotis_bbox[3])  # Нижняя граница Carotis
                crop_bottom = image_height  # Высота изображения
            else:
                print(f"Carotis и Node пересекаются в файле {annotation_file}. Пропускаю обработку.")
                skipped_files_count += 1  # Увеличиваем счетчик пропущенных файлов
                continue

        mask = Image.open(node_mask_path)

        if 'crop_left' in locals() and 'crop_right' in locals():
            cropped_image = image.crop((crop_left, 0, crop_right, image_height))
            cropped_mask = mask.crop((crop_left, 0, crop_right, mask.height))
            crop_offset_x = crop_left
            crop_offset_y = 0
        elif 'crop_top' in locals() and 'crop_bottom' in locals():
            cropped_image = image.crop((0, crop_top, image_width, crop_bottom))
            cropped_mask = mask.crop((0, crop_top, mask.width, crop_bottom))
            crop_offset_x = 0
            crop_offset_y = crop_top

        cropped_width = cropped_image.width
        cropped_height = cropped_image.height

        if cropped_width == 0 or cropped_height == 0:
            raise ValueError("Ширина или высота обрезанного изображения равна нулю. Проверьте bounding box Carotis.")

        scale_factor = 528 / max(cropped_width, cropped_height)

        new_width = int(cropped_width * scale_factor)
        new_height = int(cropped_height * scale_factor)
        stretched_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        stretched_mask = cropped_mask.resize((new_width, new_height), Image.Resampling.NEAREST)

        padding_top = (528 - new_height) // 2
        padding_bottom = 528 - new_height - padding_top
        padding_left = (528 - new_width) // 2
        padding_right = 528 - new_width - padding_left

        final_image = Image.new("RGB", (528, 528), color=(0, 0, 0))
        final_mask = Image.new("L", (528, 528), color=0)

        if 'crop_left' in locals() and 'crop_right' in locals():
            final_image.paste(stretched_image, (padding_left, padding_top))
            final_mask.paste(stretched_mask, (padding_left, padding_top))
        elif 'crop_top' in locals() and 'crop_bottom' in locals():
            final_image.paste(stretched_image, (padding_left, padding_top))
            final_mask.paste(stretched_mask, (padding_left, padding_top))

        output_image_path = os.path.join(output_images_dir, os.path.basename(image_path))
        output_mask_path = os.path.join(output_masks_dir, os.path.basename(node_mask_path))
        final_image.save(output_image_path)
        final_mask.save(output_mask_path)

        # Пересчет координат для Node
        if 'crop_left' in locals() and 'crop_right' in locals():
            new_node_bbox = [
                ((node_bbox[0] - crop_offset_x) * scale_factor) + padding_left,  # Новый x (с учетом обрезки и padding)
                (node_bbox[1] * scale_factor) + padding_top,  # Новый y (с учетом padding)
                node_bbox[2] * scale_factor,  # Новая ширина
                node_bbox[3] * scale_factor  # Новая высота
            ]
        elif 'crop_top' in locals() and 'crop_bottom' in locals():
            new_node_bbox = [
                (node_bbox[0] * scale_factor) + padding_left,  # Новый x (с учетом padding)
                ((node_bbox[1] - crop_offset_y) * scale_factor) + padding_top,  # Новый y (с учетом обрезки и padding)
                node_bbox[2] * scale_factor,  # Новая ширина
                node_bbox[3] * scale_factor  # Новая высота
            ]

        print("New Node BBox:", new_node_bbox)

        # Обновление аннотации с новыми координатами
        for annotation in data["annotations"]:
            if annotation["category_id"] == 1:
                annotation["bbox"] = new_node_bbox

        output_annotation_path = os.path.join(output_annotations_dir, annotation_file)
        with open(output_annotation_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Обновленная аннотация сохранена: {output_annotation_path}")

# Вывод количества пропущенных файлов
print(f"Количество пропущенных файлов из-за пересечения Carotis и Node: {skipped_files_count}")