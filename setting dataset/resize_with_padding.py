import cv2
import numpy as np
import os


def resize_with_padding(image, target_size=(528, 528), pad_color=(0, 0, 0)):
    old_h, old_w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / old_w, target_h / old_h)
    new_w, new_h = int(old_w * scale), int(old_h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded


input_folder = "dataset_coco_neuro_2/masks"
output_folder = "dataset_coco_neuro_2_resized/masks"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png")):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        resized_image = resize_with_padding(image, target_size=(528, 528))
        output_path = os.path.join(output_folder, filename)  # Save with the same name
        cv2.imwrite(output_path, resized_image)

print("Images resized and saved successfully!")
