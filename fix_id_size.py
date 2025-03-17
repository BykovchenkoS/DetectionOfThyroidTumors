import json
import os

input_annotation_dir = "dataset_node/ann"
output_annotation_dir = "dataset_node/ann_fixed"

os.makedirs(output_annotation_dir, exist_ok=True)

FIXED_WIDTH = 528
FIXED_HEIGHT = 528

annotation_files = [f for f in os.listdir(input_annotation_dir) if f.endswith(".json")]

for annotation_file in annotation_files:
    annotation_path = os.path.join(input_annotation_dir, annotation_file)

    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    category_mapping = {}
    for i, category in enumerate(annotation_data["categories"], start=1):
        old_id = category["id"]
        category["id"] = i
        category_mapping[old_id] = i

    for ann in annotation_data["annotations"]:
        old_category_id = ann["category_id"]
        ann["category_id"] = category_mapping.get(old_category_id, old_category_id)

    for i, ann in enumerate(annotation_data["annotations"], start=1):
        ann["id"] = i

    image_info = annotation_data["images"][0]
    image_info["width"] = FIXED_WIDTH
    image_info["height"] = FIXED_HEIGHT

    output_annotation_path = os.path.join(output_annotation_dir, annotation_file)
    with open(output_annotation_path, "w") as f:
        json.dump(annotation_data, f, indent=4)

    print(f"Исправленный файл сохранен в {output_annotation_path}")