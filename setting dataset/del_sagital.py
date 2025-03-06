import os
import json


def remove_category(input_dir, output_dir, category_to_remove_id):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            with open(input_file, "r") as f:
                data = json.load(f)

            print(f"Обрабатывается файл: {input_file}")

            remaining_categories = [cat for cat in data["categories"] if cat["id"] != category_to_remove_id]
            new_category_ids = {old_cat["id"]: new_id for new_id, old_cat in enumerate(remaining_categories)}

            for new_id, old_cat in enumerate(remaining_categories):
                old_cat["id"] = new_id
            data["categories"] = remaining_categories

            updated_annotations = []
            for ann in data["annotations"]:
                old_category_id = ann["category_id"]
                if old_category_id in new_category_ids:
                    ann["category_id"] = new_category_ids[old_category_id]
                    ann["id"] = ann["category_id"]
                    updated_annotations.append(ann)

            data["annotations"] = updated_annotations

            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Файл сохранён: {output_file}\n")


input_dir = "../dataset_coco_neuro_1/val/annotations"
output_dir = "updated_annotations"
category_to_remove_id = 0

remove_category(input_dir, output_dir, category_to_remove_id)