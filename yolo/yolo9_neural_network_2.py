from ultralytics import YOLO

model = YOLO('yolov9m.pt')

try:
    model.train(data='data_for_yolo.yaml',
                epochs=100,
                batch=16,
                imgsz=640,
                lr0=0.01,
                optimizer='SGD')
except RuntimeError as e:
    print(f"Ошибка во время обучения: {e}")

print("Обучение завершено.")