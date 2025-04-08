from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def add_padding_to_image(image_path, output_path, target_size):
    img = Image.open(image_path)
    original_width, original_height = img.size

    pad_left = (target_size[0] - original_width) // 2
    pad_right = target_size[0] - original_width - pad_left
    pad_top = (target_size[1] - original_height) // 2
    pad_bottom = target_size[1] - original_height - pad_top

    padded_img = Image.new("RGB", target_size, color=(0, 0, 0))
    padded_img.paste(img, (pad_left, pad_top))

    padded_img.save(output_path)

    return pad_left, pad_top


def update_annotations(annotations_data, pad_left, pad_top):
    for annotation in annotations_data["annotations"]:
        bbox = annotation["bbox"]
        bbox[0] += pad_left  # x
        bbox[1] += pad_top   # y
    return annotations_data


def visualize_and_save_bboxes(image_path, annotations_data, output_visualization_path):
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for annotation in annotations_data["annotations"]:
        bbox = annotation["bbox"]
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(output_visualization_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def process_all_images_and_annotations(images_dir, annotations_dir, output_images_dir, output_annotations_dir, output_visualizations_dir, target_size):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)
    os.makedirs(output_visualizations_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)

        file_name = os.path.splitext(image_file)[0]

        annotation_file = f"{file_name}.json"
        annotation_path = os.path.join(annotations_dir, annotation_file)

        if not os.path.exists(annotation_path):
            print(f"Annotation file not found for {image_file}, skipping...")
            continue

        output_image_path = os.path.join(output_images_dir, image_file)
        output_annotation_path = os.path.join(output_annotations_dir, annotation_file)
        output_visualization_path = os.path.join(output_visualizations_dir, image_file)

        pad_left, pad_top = add_padding_to_image(image_path, output_image_path, target_size)

        with open(annotation_path, "r") as f:
            annotations_data = json.load(f)

        updated_annotations = update_annotations(annotations_data, pad_left, pad_top)

        with open(output_annotation_path, "w") as f:
            json.dump(updated_annotations, f, indent=4)

        visualize_and_save_bboxes(output_image_path, updated_annotations, output_visualization_path)

        print(f"Processed {image_file} and saved to {output_image_path}")


images_dir = 'dataset_sam_neuro_2/train/images/'
annotations_dir = 'dataset_sam_neuro_2/train/annotations/'
output_images_dir = 'dataset_sam_neuro_2/train/processed_images/'
output_annotations_dir = 'dataset_sam_neuro_2/train/processed_annotations/'
output_visualizations_dir = 'dataset_sam_neuro_2/train/visualized_images/'
target_size = (1024, 1024)

process_all_images_and_annotations(
    images_dir,
    annotations_dir,
    output_images_dir,
    output_annotations_dir,
    output_visualizations_dir,
    target_size
)

