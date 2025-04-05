import torch
from ultralytics import YOLO
import pandas as pd
import os
import yaml


def train_yolo():
    with open('data_for_yolo_2.yaml') as f:
        data = yaml.safe_load(f)
    class_names = data['names']

    model = YOLO('yolov9m.pt')

    results = model.train(
        data='data_for_yolo_2.yaml',
        epochs=100,
        batch=16,
        imgsz=544,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        val=True,
        save_json=True,
        plots=True
    )

    val_results = model.val(save_json=True, save_dir=results.save_dir)

    metrics_data = []
    for cls_idx, cls_name in enumerate(class_names):
        metrics_data.append({
            'class': cls_name,
            'precision': val_results.box.mp,
            'recall': val_results.box.mr,
            'mAP50': val_results.box.map50,
            'mAP50-95': val_results.box.map
        })

    df = pd.DataFrame(metrics_data)

    csv_path = os.path.join(results.save_dir, 'class_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Метрики сохранены в {csv_path}")
    print("\nРезультаты валидации:")
    print(df)


if __name__ == "__main__":
    train_yolo()
