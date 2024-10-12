import torch
from ultralytics import YOLO

device = torch.device('cpu')
model = YOLO('yolov3.pt')
model.to(device)

data_config = 'data_for_yolo.yaml'
epochs = 50
patience = 50
batch_size = 16
img_size = 640

model.train(data=data_config,
            epochs=epochs,
            patience=patience,
            batch=batch_size,
            imgsz=img_size,
            save=True,
            save_period=-1,
            cache=False,
            workers=8,
            verbose=True,
            device='cpu')

print("Обучение завершено.")
