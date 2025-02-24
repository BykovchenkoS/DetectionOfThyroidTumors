import os
import json


def replace_in_json(data, old_str, new_str):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.replace(old_str, new_str)
            elif isinstance(value, (dict, list)):
                replace_in_json(value, old_str, new_str)
    elif isinstance(data, list):
        for item in data:
            replace_in_json(item, old_str, new_str)


def process_and_save_json_file(file_path, output_folder, old_str, new_str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    replace_in_json(data, old_str, new_str)
    relative_path = os.path.relpath(file_path, start=input_folder)
    new_file_path = os.path.join(output_folder, relative_path)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def process_folder(input_folder, output_folder, old_str, new_str):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_path}")
                process_and_save_json_file(file_path, output_folder, old_str, new_str)


input_folder = "dataset_for_search_2/train/annotations"

output_folder = "new_path"

old_str = "dataset_coco_neuro_2"
new_str = "dataset_for_search_2"

process_folder(input_folder, output_folder, old_str, new_str)
