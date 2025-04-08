from PIL import Image
import os
import matplotlib.pyplot as plt


def overlay_mask_on_image(image, mask_path, color, alpha=0.5):
    mask = Image.open(mask_path).convert("L")

    if mask.size != image.size:
        print(f"Size mismatch: Image size {image.size}, Mask size {mask.size}")
        raise ValueError("Image and mask sizes do not match!")

    colored_mask = Image.new("RGBA", mask.size, color)
    colored_mask.putalpha(mask)
    overlayed_image = Image.alpha_composite(image, colored_mask)

    return overlayed_image


def process_specific_annotation(annotation_data, masks_dir):
    image_info = annotation_data["images"][0]
    image_path = image_info["file_name"]

    base_image = Image.open(image_path).convert("RGBA")
    print(f"Loaded image: {image_path}, size: {base_image.size}")

    category_colors = {
        1: (255, 0, 0, 128),  # Красный для "Thyroid tissue"
        2: (0, 0, 255, 128)   # Синий для "Carotis"
    }

    for annotation in annotation_data["annotations"]:
        if annotation["image_id"] == image_info["id"]:
            mask_file = annotation["segmentation"]
            mask_path = os.path.join(masks_dir, os.path.basename(mask_file))

            mask = Image.open(mask_path).convert("L")
            print(f"Loaded mask: {mask_path}, size: {mask.size}")

            category_id = annotation["category_id"]
            color = category_colors.get(category_id, (0, 255, 0, 128))

            base_image = overlay_mask_on_image(base_image, mask_path, color)

    plt.imshow(base_image)
    plt.axis('off')
    plt.title("Image with Overlaid Masks")
    plt.show()


annotation_data = {
    "images": [
        {
            "id": 33,
            "file_name": "dataset_sam_neuro_2/images_neuro_2/33.jpg",
            "width": 334,
            "height": 206
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 33,
            "category_id": 1,
            "segmentation": "dataset_sam_neuro_2/masks/33_Node_1558813313.png",
            "area": 241,
            "bbox": [
                459.83233532934133,
                361.6467065868263,
                94.8502994011976,
                61.65269461077845
            ],
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "Node",
            "supercategory": "organ"
        }
    ]
}


masks_dir = 'dataset_sam_neuro_2/masks/'
process_specific_annotation(annotation_data, masks_dir)
