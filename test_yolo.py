from ultralytics import YOLO
import cv2
from pathlib import Path
import os

weights_path = 'yolo/train_yolo5_2/weights/best.pt'
model = YOLO(weights_path)

images_path = 'screen foto/dataset 2024-04-21 14_33_36/images_neuro_2/train'
image_files = list(Path(images_path).glob('*.jpg'))

results_dir = 'predict_yolo_5_node'
os.makedirs(results_dir, exist_ok=True)

for image_path in image_files:
    img = cv2.imread(str(image_path))

    results = model.predict(source=img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = Path(results_dir) / f"{image_path.stem}_result.jpg"
        cv2.imwrite(str(output_path), img)
