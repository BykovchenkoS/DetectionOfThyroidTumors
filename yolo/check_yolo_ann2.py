import cv2
import os

IMG_DIR = "../screen foto/dataset 2024-04-21 14_33_36/img"
YOLO2_DIR = "../screen foto/dataset 2024-04-21 14_33_36/yolo2_ann"

annotation_files = [f for f in os.listdir(YOLO2_DIR) if f.endswith('.txt')]

for annotation_filename in annotation_files:
    annotation_path = os.path.join(YOLO2_DIR, annotation_filename)
    img_filename = annotation_filename.replace('.txt', '.jpg')
    img_path = os.path.join(IMG_DIR, img_filename)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        continue

    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    for annotation in annotations:
        parts = annotation.strip().split()
        class_id = int(parts[0])  # ID класса
        x_center, y_center, w, h = map(float, parts[1:])

        img_height, img_width = img.shape[:2]
        x_min = int((x_center - w / 2) * img_width)
        y_min = int((y_center - h / 2) * img_height)
        x_max = int((x_center + w / 2) * img_width)
        y_max = int((y_center + h / 2) * img_height)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow(f"Annotated Image - {img_filename}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
