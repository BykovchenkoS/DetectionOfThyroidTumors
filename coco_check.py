import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from skimage import measure

data = {
    "images": [
        {
            "id": 100,
            "file_name": "screen foto/dataset 2024-04-21 14_33_36/images_neuro_1/train/100.jpg",
            "width": 375,
            "height": 500
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 100,
            "category_id": 0,
            "segmentation": [
                [
                    0,
                    180
                ],
                [
                    0,
                    416
                ],
                [
                    357,
                    406
                ],
                [
                    364,
                    175
                ]
            ],
            "area": 84150,
            "bbox": [
                0,
                175,
                364,
                241
            ],
            "iscrowd": 0
        },
        {
            "id": 1,
            "image_id": 100,
            "category_id": 1,
            "segmentation": "screen foto/dataset 2024-04-21 14_33_36/masks/100_Thyroid_tissue_1554990403.png",
            "area": 577,
            "bbox": [
                155,
                254,
                92,
                76
            ],
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 100,
            "category_id": 2,
            "segmentation": "screen foto/dataset 2024-04-21 14_33_36/masks/100_Carotis_1554990481.png",
            "area": 583,
            "bbox": [
                243,
                265,
                34,
                41
            ],
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "sagital_longitudinal",
            "supercategory": "organ"
        },
        {
            "id": 1,
            "name": "Thyroid tissue",
            "supercategory": "organ"
        },
        {
            "id": 2,
            "name": "Carotis",
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
        # Если аннотация использует сегментацию в виде полигонов
        polygon = ann["segmentation"]
        poly = np.array(polygon)
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.3, color=color)

    if isinstance(ann["segmentation"], str):
        # Если аннотация использует битмап
        mask_path = ann["segmentation"]
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = mask > 127

        contours = measure.find_contours(mask, 0.5)
        x, y, w, h = ann["bbox"]
        for contour in contours:
            contour[:, 0] += y
            contour[:, 1] += x
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

    x, y, w, h = ann["bbox"]
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

plt.axis()
plt.show()
