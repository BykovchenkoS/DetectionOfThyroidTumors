import os
import json


def shift_category_ids_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            annotation_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            with open(annotation_file, 'r') as f:
                data = json.load(f)

            old_to_new_category_id = {}
            for category in data['categories']:
                old_id = category['id']
                new_id = old_id + 1
                category['id'] = new_id
                old_to_new_category_id[old_id] = new_id

            for annotation in data['annotations']:
                old_category_id = annotation['category_id']
                annotation['category_id'] = old_to_new_category_id[old_category_id]

            for i, annotation in enumerate(data['annotations']):
                annotation['id'] = i + 1

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)

            print(f"Обновленные аннотации сохранены в файл: {output_file}")


input_dir = 'dataset_coco_neuro_2/train/annotations'
output_dir = 'updated_annotations'

shift_category_ids_in_directory(input_dir, output_dir)