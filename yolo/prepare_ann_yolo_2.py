import json
import os
import cv2
from tqdm import tqdm


def convert_coco_to_yolo(coco_json_path, output_dir):
    """Преобразует COCO аннотации в YOLO формат с проверкой данных"""
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки JSON {coco_json_path}: {str(e)}")
        return

    if 'images' not in coco_data or 'annotations' not in coco_data or 'categories' not in coco_data:
        print(f"Файл {coco_json_path} не содержит полных COCO данных")
        return

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    if not categories:
        print(f"Нет категорий в файле {coco_json_path}")
        return

    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    images = {img['id']: img for img in coco_data['images']}
    if not images:
        print(f"Нет изображений в файле {coco_json_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0

    for image_id, img_info in images.items():
        annotations = annotations_by_image.get(image_id, [])
        if not annotations:
            continue

        base_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        yolo_txt_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(yolo_txt_path, 'w') as f:
            for ann in annotations:
                if 'bbox' not in ann or len(ann['bbox']) != 4:
                    continue

                category_id = ann['category_id']
                x, y, w, h = ann['bbox']

                image_width = img_info.get('width', 528)
                image_height = img_info.get('height', 528)

                if w <= 0 or h <= 0:
                    continue

                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                width_norm = w / image_width
                height_norm = h / image_height

                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    continue
                if not (0 < width_norm <= 1 and 0 < height_norm <= 1):
                    continue

                f.write(f"{category_id - 1} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                processed_count += 1

    print(f"Обработано {processed_count} аннотаций из {len(coco_data['annotations'])} для {len(images)} изображений")


def visualize_yolo_annotations(image_path, annotation_path, classes, output_dir=None):
    """Визуализирует YOLO-разметку на изображении с проверкой данных"""
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return

    img_height, img_width = image.shape[:2]

    if not os.path.exists(annotation_path):
        print(f"Файл аннотаций не найден: {annotation_path}")
        return

    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    if not annotations:
        print(f"Пустой файл аннотаций: {annotation_path}")
        return

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
        except ValueError:
            continue

        x = int((x_center - width / 2) * img_width)
        y = int((y_center - height / 2) * img_height)
        w = int(width * img_width)
        h = int(height * img_height)

        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))

        color = colors[class_id % len(colors)]
        class_name = classes[class_id] if class_id < len(classes) else str(class_id)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, class_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        print(f"Визуализация сохранена: {output_path}")
    else:
        cv2.imshow("YOLO Annotation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_all_annotations(annotations_dir, images_dir, yolo_output_dir, viz_output_dir):
    """Обрабатывает все аннотации в директории с подробным логированием"""
    print(f"Начало обработки аннотаций из {annotations_dir}")

    classes = []
    for json_file in os.listdir(annotations_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(annotations_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    coco_data = json.load(f)
                classes = [cat['name'] for cat in coco_data.get('categories', [])]
                break
            except Exception as e:
                print(f"Ошибка загрузки {json_path}: {str(e)}")
                continue

    if not classes:
        print("Не удалось определить категории. Проверьте файлы аннотаций.")
        return

    print(f"Найдены категории: {', '.join(classes)}")
    os.makedirs(yolo_output_dir, exist_ok=True)
    os.makedirs(viz_output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    if not json_files:
        print(f"Не найдено JSON файлов в {annotations_dir}")
        return

    for json_file in tqdm(json_files, desc="Обработка аннотаций"):
        json_path = os.path.join(annotations_dir, json_file)
        print(f"\nОбработка файла: {json_path}")

        convert_coco_to_yolo(json_path, yolo_output_dir)

        try:
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки {json_path}: {str(e)}")
            continue

        if 'images' not in coco_data:
            print(f"Файл {json_path} не содержит информации об изображениях")
            continue

        for img_info in coco_data['images']:
            if 'file_name' not in img_info:
                continue

            image_path = os.path.join(images_dir, os.path.basename(img_info['file_name']))
            if not os.path.exists(image_path):
                print(f"Изображение не найдено: {image_path}")
                continue

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
    annotations_dir = "dataset_coco_neuro_3/train/annotations"
    images_dir = "dataset_coco_neuro_3/train/images"
    yolo_output_dir = "dataset_yolo_neuro_2/labels/train"
    viz_output_dir = "yolo_visualized_annotations"

    if not os.path.exists(annotations_dir):
        print(f"Директория с аннотациями не найдена: {annotations_dir}")
    elif not os.path.exists(images_dir):
        print(f"Директория с изображениями не найдена: {images_dir}")
    else:
        process_all_annotations(
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            yolo_output_dir=yolo_output_dir,
            viz_output_dir=viz_output_dir
        )

    print("Обработка завершена.")
