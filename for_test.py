import os
import cv2
from collections import Counter
import matplotlib.pyplot as plt


"""
    Проверим корректности разметки и провизуализируем результаты
"""

images_path = 'screen foto/dataset 2024-04-21 14_33_36/img'
labels_path = 'screen foto/dataset 2024-04-21 14_33_36/yolo_labels'

class_names = {
    0: 'sagital',
    1: 'longitudinal',
    2: 'Thyroid tissue',
    3: 'Node',
    4: 'Carotis',
    5: 'Jugular'
}

# for image_file in os.listdir(images_path):
#     image_path = os.path.join(images_path, image_file)
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Не удалось загрузить изображение: {image_file}")
#         continue
#
#     label_file = os.path.join(labels_path, f"{os.path.splitext(image_file)[0]}.txt")
#
#     if os.path.exists(label_file):
#         with open(label_file, 'r') as f:
#             for line in f.readlines():
#                 class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
#                 class_name = class_names.get(int(class_id), 'Unknown')
#
#                 #конвертация координат в пиксели
#                 img_height, img_width = img.shape[:2]
#                 x_center *= img_width
#                 y_center *= img_height
#                 bbox_width *= img_width
#                 bbox_height *= img_height
#
#                 #определение углов прямоугольника
#                 x1 = int(x_center - bbox_width / 2)
#                 y1 = int(y_center - bbox_height / 2)
#                 x2 = int(x_center + bbox_width / 2)
#                 y2 = int(y_center + bbox_height / 2)
#
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         annotation_text = f"Image: {image_file}, Annotation: {os.path.basename(label_file)}"
#         cv2.putText(img, annotation_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#         cv2.imshow('Annotated Image', img)
#         key = cv2.waitKey(0)
#
#         if key == ord('q'):
#             break
#
#     else:
#         print(f"Аннотация отсутствует для изображения: {image_file}")

# cv2.destroyAllWindows()

"""
Проверим равномерность распределения классов
"""

labels_path = 'screen foto/dataset 2024-04-21 14_33_36/yolo_labels'
class_counts = Counter()

for label_file in os.listdir(labels_path):
    if label_file.endswith('.txt'):
        label_path = os.path.join(labels_path, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

for class_id, count in class_counts.items():
    print(f"{class_names.get(class_id, 'Unknown')}: {count}")

labels = [class_names.get(class_id, 'Unknown') for class_id in class_counts.keys()]
counts = [count for count in class_counts.values()]

plt.figure(figsize=(10, 5))
plt.bar(labels, counts, color='blue')
plt.xlabel('Классы')
plt.ylabel('Количество объектов')
plt.title('Распределение классов в наборе данных')
plt.xticks(rotation=45)
plt.show()