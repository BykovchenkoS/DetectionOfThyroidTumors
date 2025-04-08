import json
import os
import cv2


def convert_coco_to_yolo(coco_json_path, output_dir):
    """Преобразует COCO аннотации в YOLO формат"""
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    annotations_by_image = {}

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    images = {img['id']: img['file_name'] for img in coco_data['images']}
    os.makedirs(output_dir, exist_ok=True)

    for image_id, file_name in images.items():
        annotations = annotations_by_image.get(image_id, [])
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        yolo_txt_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(yolo_txt_path, 'w') as f:
            for ann in annotations:
                category_id = ann['category_id']
                x, y, w, h = ann['bbox']

                image_width = 528
                image_height = 528

                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                width_norm = w / image_width
                height_norm = h / image_height

                f.write(f"{category_id - 1} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")


def visualize_yolo_annotations(image_path, annotation_path, classes, output_dir=None):
    """Визуализирует YOLO-разметку на изображении"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return

    img_height, img_width = image.shape[:2]

    if not os.path.exists(annotation_path):
        print(f"Файл аннотаций {annotation_path} не найден")
        return

    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])

        x = int((x_center - width / 2) * img_width)
        y = int((y_center - height / 2) * img_height)
        w = int(width * img_width)
        h = int(height * img_height)

        color = colors[class_id % len(colors)]
        class_name = classes[class_id] if class_id < len(classes) else str(class_id)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, class_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
    else:
        cv2.imshow("YOLO Annotation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_all_annotations(annotations_dir, images_dir, yolo_output_dir, viz_output_dir):
    """Обрабатывает все аннотации в директории"""
    classes = ["Thyroid tissue", "Carotis"]

    os.makedirs(yolo_output_dir, exist_ok=True)
    os.makedirs(viz_output_dir, exist_ok=True)

    for json_file in os.listdir(annotations_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(annotations_dir, json_file)

            convert_coco_to_yolo(json_path, yolo_output_dir)

            with open(json_path, 'r') as f:
                coco_data = json.load(f)

            for img_info in coco_data['images']:
                image_path = os.path.join(images_dir, os.path.basename(img_info['file_name']))
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                yolo_txt_path = os.path.join(yolo_output_dir, f"{base_name}.txt")

                if os.path.exists(yolo_txt_path):
                    visualize_yolo_annotations(
                        image_path=image_path,
                        annotation_path=yolo_txt_path,
                        classes=classes,
                        output_dir=viz_output_dir
                    )


if __name__ == "__main__":
    annotations_dir = "dataset_coco_neuro_1/val/annotations"
    images_dir = "dataset_coco_neuro_1/val/images"
    yolo_output_dir = "dataset_yolo_neuro_1/labels/val"
    viz_output_dir = "yolo_visualized_annotations"

    process_all_annotations(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        yolo_output_dir=yolo_output_dir,
        viz_output_dir=viz_output_dir
    )

    print("Обработка всех аннотаций завершена.")
