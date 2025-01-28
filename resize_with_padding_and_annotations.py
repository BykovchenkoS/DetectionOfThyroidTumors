import cv2
import os
import json


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

    return padded, scale, scale, left, top


input_folder = "dataset_coco_neuro_2_NO_resized/val/images"
output_folder = "dataset_coco_neuro_2/val/images"
input_annotations_folder = "dataset_coco_neuro_2_NO_resized/val/annotations"
output_annotations_folder = "dataset_coco_neuro_2/val/annotations"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_annotations_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        annotation_path = os.path.join(input_annotations_folder, f"{os.path.splitext(filename)[0]}.json")

        if not os.path.exists(annotation_path):
            print(f"Annotation not found for image {filename}, skipping...")
            continue

        image = cv2.imread(img_path)
        resized_image, scale_w, scale_h, offset_x, offset_y = resize_with_padding(image, target_size=(528, 528))
        output_img_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_img_path, resized_image)

        with open(annotation_path, "r") as f:
            annotation_data = json.load(f)

        for annotation in annotation_data["annotations"]:
            bbox = annotation["bbox"]

            new_bbox = [
                (bbox[0] + offset_x) * scale_w,  # x
                (bbox[1] * scale_h) + offset_y,  # y
                bbox[2] * scale_w,  # width
                bbox[3] * scale_h  # height
            ]
            annotation["bbox"] = new_bbox

            if isinstance(annotation["segmentation"], list):
                new_segmentation = []
                for segment in annotation["segmentation"]:
                    updated_segment = [
                        (segment[i] + offset_x) * scale_w if i % 2 == 0 else (segment[i] + offset_y) * scale_h
                        for i in range(len(segment))
                    ]
                    new_segmentation.append(updated_segment)
                annotation["segmentation"] = new_segmentation
            elif isinstance(annotation["segmentation"], str):
                mask_path = os.path.join(os.path.dirname(annotation_path), annotation["segmentation"])
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    resized_mask, _, _, _, _ = resize_with_padding(mask, target_size=(528, 528))
                    output_mask_path = os.path.join(output_annotations_folder, os.path.basename(mask_path))
                    cv2.imwrite(output_mask_path, resized_mask)
                    annotation["segmentation"] = os.path.relpath(output_mask_path, output_annotations_folder)

        output_annotation_path = os.path.join(output_annotations_folder, f"{os.path.splitext(filename)[0]}.json")
        with open(output_annotation_path, "w") as f:
            json.dump(annotation_data, f, indent=4)

print("Images and annotations resized and saved successfully")
