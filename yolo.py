from ultralytics import YOLO

model = YOLO('yolov3.pt')

try:
    model.train(data='data_for_yolo.yaml',
                epochs=30,
                batch=8,
                imgsz=416,
                lr0=0.01,
                optimizer='SGD')
except RuntimeError as e:
    print(f"Ошибка во время обучения: {e}")

print("Обучение завершено.")