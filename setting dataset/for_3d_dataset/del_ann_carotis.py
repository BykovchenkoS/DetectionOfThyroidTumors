import json
import os

input_annotations_dir = "../../dataset_coco_neuro_3/ann"
output_annotations_dir = "dataset_coco_neuro_3/ann_new"

os.makedirs(output_annotations_dir, exist_ok=True)

for annotation_file in os.listdir(input_annotations_dir):
    if not annotation_file.endswith(".json"):
        continue

    input_json_path = os.path.join(input_annotations_dir, annotation_file)
    output_json_path = os.path.join(output_annotations_dir, annotation_file)

    print(f"Обработка файла: {annotation_file}")

    with open(input_json_path, "r") as f:
        data = json.load(f)

    carotis_found = any(category["id"] == 2 for category in data["categories"])

    if carotis_found:
        data["categories"] = [category for category in data["categories"] if category["id"] != 2]

        data["annotations"] = [annotation for annotation in data["annotations"] if annotation["category_id"] != 2]

    for image in data["images"]:
        if "file_name" in image:
            image["file_name"] = image["file_name"].replace("dataset_coco_neuro_2", "dataset_coco_neuro_3")
            image["file_name"] = image["file_name"].replace("images_neuro_2", "images_neuro_3")

    for annotation in data["annotations"]:
        if "segmentation" in annotation and isinstance(annotation["segmentation"], str):
            annotation["segmentation"] = annotation["segmentation"].replace("dataset_coco_neuro_2", "dataset_coco_neuro_3")
            annotation["segmentation"] = annotation["segmentation"].replace("images_neuro_2", "images_neuro_3")

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Обновленный JSON-файл сохранен: {output_json_path}")