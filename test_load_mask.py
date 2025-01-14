import torch
import numpy as np
import cv2
from PIL import Image


def create_mask_from_segmentation(annotations, img_width, img_height):
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for ann in annotations:
        segmentation = ann['segmentation']

        if isinstance(segmentation, list):
            segmentation = np.array(segmentation, dtype=np.int32)
            polygon = segmentation.reshape(-1, 2)
            cv2.fillPoly(mask, [polygon], 1)

        elif isinstance(segmentation, str):
            mask_image = Image.open(segmentation).convert('L')
            mask_image = np.array(mask_image)
            mask_image = np.where(mask_image > 0, 1, 0)
            mask = np.maximum(mask, mask_image)

        else:
            print(f"Не поддерживаемый формат сегментации: {segmentation}")

    return torch.as_tensor(mask, dtype=torch.uint8)


def convert_masks_to_tensor(img_width, img_height, annotations):
    masks = []
    for ann in annotations:
        mask = create_mask_from_segmentation([ann], img_width, img_height)
        masks.append(mask)

    return masks


annotations = [
    {
        "id": 0,
        "image_id": 2,
        "category_id": 0,
        "segmentation": [
            [16, 55],
            [11, 329],
            [401, 323],
            [392, 53]
        ],
        "area": 104184,
        "bbox": [11, 53, 390, 276],
        "iscrowd": 0
    },
    {
        "id": 1,
        "image_id": 2,
        "category_id": 1,
        "segmentation": "screen foto/dataset 2024-04-21 14_33_36/masks/2_Thyroid_tissue_1556365511.png",
        "area": 651,
        "bbox": [67, 68, 324, 192],
        "iscrowd": 0
    },
    {
        "id": 2,
        "image_id": 2,
        "category_id": 2,
        "segmentation": "screen foto/dataset 2024-04-21 14_33_36/masks/2_Carotis_1556365529.png",
        "area": 311,
        "bbox": [12, 145, 87, 67],
        "iscrowd": 0
    }
]

img_width = 500
img_height = 377

masks = convert_masks_to_tensor(img_width, img_height, annotations)

for i, mask in enumerate(masks):
    torch.save(mask, f'mask_{i}.pt')
    np.save(f'mask_{i}.npy', mask.numpy())
    print(f"Маска {i} сохранена в файл.")
