import os
import shutil

jpg_folder = 'dataset_coco_neuro_2/val/images'
json_folder = 'ann_new_2_node'
destination_folder = 'dataset_coco_neuro_2/val/annotations'

jpg_files = {os.path.splitext(file)[0] for file in os.listdir(jpg_folder) if file.endswith('.jpg')}
txt_files = {os.path.splitext(file)[0] for file in os.listdir(json_folder) if file.endswith('.json')}

common_files = jpg_files & txt_files

for file in common_files:
    txt_file_path = os.path.join(json_folder, f"{file}.json")
    shutil.copy(txt_file_path, destination_folder)

print(f"Скопировано {len(common_files)} .json файлов.")
