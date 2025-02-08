import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from skimage import measure

data = {
    "images": [
        {
            "id": 6,
            "file_name": "dataset_coco_neuro_2/images_neuro_2/6.jpg",
            "width": 171,
            "height": 91
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 6,
            "category_id": 0,
            "segmentation": "dataset_coco_neuro_2/masks/6_Node_1557921606.png",
            "area": 198,
            "bbox": [
                98.80701754385964,
                173.4035087719298,
                296.42105263157896,
                166.73684210526315
            ],
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "Node",
            "supercategory": "organ"
        }
    ]
}


category_colors = {
    0: (0.8, 0.2, 0.2),  # sagital_longitudinal - красный
    1: (0.2, 0.8, 0.2),  # Thyroid tissue - зеленый
    2: (0.2, 0.2, 0.8),  # Carotis - синий
}

image_path = data["images"][0]["file_name"]
img = Image.open(image_path)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)

for ann in data["annotations"]:
    category_id = ann["category_id"]
    color = category_colors.get(category_id, (1, 1, 1))

    if isinstance(ann["segmentation"], list):
        polygon = ann["segmentation"]
        poly = np.array(polygon)
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.3, color=color)

    if isinstance(ann["segmentation"], str):
        mask_path = ann["segmentation"]
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = mask > 127

        contours = measure.find_contours(mask, 0.5)
        x, y, w, h = ann["bbox"]
        for contour in contours:
            # contour[:, 0] += y
            # contour[:, 1] += x
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

    x, y, w, h = ann["bbox"]
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

plt.axis()
plt.show()
