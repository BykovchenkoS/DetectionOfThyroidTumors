import json
import os


def update_paths_in_annotations(annotation_path, output_annotation_path):
    with open(annotation_path, "r") as f:
        annotations_data = json.load(f)

    # Обновляем пути в поле "file_name"
    for image in annotations_data["images"]:
        if "file_name" in image:
            image["file_name"] = image["file_name"].replace("dataset_coco_neuro", "dataset_sam_neuro")

    # Обновляем пути в поле "segmentation"
    for annotation in annotations_data["annotations"]:
        if "segmentation" in annotation:
            annotation["segmentation"] = annotation["segmentation"].replace("dataset_coco_neuro", "dataset_sam_neuro")

    with open(output_annotation_path, "w") as f:
        json.dump(annotations_data, f, indent=4)

    print(f"Updated paths in {annotation_path} and saved to {output_annotation_path}")


def process_all_annotations(annotations_dir, output_annotations_dir):
    os.makedirs(output_annotations_dir, exist_ok=True)
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]

    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotations_dir, annotation_file)
        output_annotation_path = os.path.join(output_annotations_dir, annotation_file)
        update_paths_in_annotations(annotation_path, output_annotation_path)


annotations_dir = 'dataset_sam_neuro_1/val/annotations/'
output_annotations_dir = 'dataset_sam_neuro_1/val/annotations_new/'

process_all_annotations(annotations_dir, output_annotations_dir)
