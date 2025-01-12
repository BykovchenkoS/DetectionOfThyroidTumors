import cv2
import os


def draw_yolo_bboxes(image_path, yolo_annotation_path, output_dir):
    if not os.path.exists(image_path):
        print(f"Изображение {image_path} не найдено.")
        return

    if not os.path.exists(yolo_annotation_path):
        print(f"Файл разметки {yolo_annotation_path} не найден.")
        return

    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    with open(yolo_annotation_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        x_center = int(x_center * img_width)
        y_center = int(y_center * img_height)
        width = int(width * img_width)
        height = int(height * img_height)

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Class: {class_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Изображение с bbox сохранено: {output_path}")


def visualize_yolo_annotations(image_dir, yolo_dir, output_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0].replace('cropped_', '')
        image_path = os.path.join(image_dir, image_file)
        yolo_annotation_path = os.path.join(yolo_dir, f"{base_name}.txt")

        draw_yolo_bboxes(image_path, yolo_annotation_path, output_dir)


if __name__ == "__main__":
    image_dir = "../screen foto/dataset 2024-04-21 14_33_36/img_masks_thyroid_carotis/"
    yolo_dir = "screen foto/dataset 2024-04-21 14_33_36/yolo_annotations/"
    output_dir = "output_visualized_labels_for_yolo2/"

    visualize_yolo_annotations(image_dir, yolo_dir, output_dir)
