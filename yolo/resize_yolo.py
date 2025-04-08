import cv2
import numpy as np
import os
from tqdm import tqdm


def add_padding_and_visualize(image_dir, labels_dir, output_image_dir, output_labels_dir, visualization_dir,
                              target_size=544):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(image_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        delta_w = target_size - w
        delta_h = target_size - h
        top = bottom = delta_h // 2
        left = right = delta_w // 2

        if delta_h % 2 != 0:
            bottom += 1
        if delta_w % 2 != 0:
            right += 1

        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))

        output_img_path = os.path.join(output_image_dir, img_name)
        cv2.imwrite(output_img_path, padded_img)

        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)
        output_txt_path = os.path.join(output_labels_dir, txt_name)

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f_in, open(output_txt_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_center, y_center, width, height = map(float, parts)

                    new_x = (x_center * w + left) / target_size
                    new_y = (y_center * h + top) / target_size
                    new_w = (width * w) / target_size
                    new_h = (height * h) / target_size

                    f_out.write(f"{int(class_id)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")

                    x1 = int((new_x - new_w / 2) * target_size)
                    y1 = int((new_y - new_h / 2) * target_size)
                    x2 = int((new_x + new_w / 2) * target_size)
                    y2 = int((new_y + new_h / 2) * target_size)

                    color = (0, 255, 0) if int(class_id) == 0 else (0, 0, 255)
                    cv2.rectangle(padded_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(padded_img, f"Class {int(class_id)}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            vis_path = os.path.join(visualization_dir, img_name)
            cv2.imwrite(vis_path, padded_img)


add_padding_and_visualize(
    image_dir='dataset_yolo_neuro_2/images/train',
    labels_dir='dataset_yolo_neuro_2/labels/train',
    output_image_dir='dataset_yolo_neuro_2/images/train_new',
    output_labels_dir='dataset_yolo_neuro_2/labels/train_new',
    visualization_dir='visualization_val_new'
)
