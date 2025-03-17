import cv2
import json
import os
import matplotlib.pyplot as plt


def visualize_annotations(image_path, annotation_path):
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        return

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    for ann in annotation_data["annotations"]:
        category_id = ann["category_id"]
        bbox = ann["bbox"]

        if category_id == 1:  # Node
            color = (255, 0, 0)  # Красный
        elif category_id == 2:  # Carotis
            color = (0, 255, 0)  # Зеленый
        else:
            color = (0, 0, 255)  # Синий

        x_min, y_min, width, height = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), color, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Visualization of Bounding Boxes")
    plt.axis("off")
    plt.show()


input_annotation_dir_1 = "dataset_node/ann_fixed"
input_image_dir_1 = "dataset_node/img"

while True:
    annotation_number = input("Введите номер аннотации (или 'exit' для выхода): ").strip()

    if annotation_number.lower() == "exit":
        print("Программа завершена.")
        break

    annotation_file = f"{annotation_number}.json"
    annotation_path = os.path.join(input_annotation_dir_1, annotation_file)

    if not os.path.exists(annotation_path):
        print(f"Ошибка: Аннотация '{annotation_file}' не найдена в директории '{input_annotation_dir_1}'.")
        continue

    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    image_file_name = annotation_data["images"][0]["file_name"]
    image_file_name = os.path.basename(image_file_name)
    image_path = os.path.join(input_image_dir_1, image_file_name)

    if not os.path.exists(image_path):
        print(f"Ошибка: Изображение '{image_file_name}' не найдено в директории '{input_image_dir_1}'.")
        continue

    print(f"Визуализация для файла: {annotation_file}")
    visualize_annotations(image_path, annotation_path)
