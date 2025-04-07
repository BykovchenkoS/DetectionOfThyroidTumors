import torch
from ultralytics import YOLO
import pandas as pd
import os
import numpy as np
import yaml


class ClassMetricsLogger:
    def __init__(self, class_names):
        self.class_names = class_names
        self.metrics_history = {
            'epoch': [],
            'cls_precision': {name: [] for name in class_names},
            'cls_recall': {name: [] for name in class_names},
            'cls_map50': {name: [] for name in class_names},
            'cls_map50_95': {name: [] for name in class_names}
        }
        self.current_epoch = 0

    def update(self, validator):
        """Обновляет метрики после каждой эпохи"""
        self.current_epoch += 1
        self.metrics_history['epoch'].append(self.current_epoch)

        if hasattr(validator, 'metrics'):
            for cls_idx, cls_name in enumerate(self.class_names):
                try:
                    cls_result = validator.metrics.class_result(cls_idx)
                    cls_p, cls_r, cls_map50, cls_map = cls_result

                    self.metrics_history['cls_precision'][cls_name].append(cls_p)
                    self.metrics_history['cls_recall'][cls_name].append(cls_r)
                    self.metrics_history['cls_map50'][cls_name].append(cls_map50)
                    self.metrics_history['cls_map50_95'][cls_name].append(cls_map)
                except Exception as e:
                    print(f"Ошибка при получении метрик для класса {cls_name}: {e}")

                    self.metrics_history['cls_precision'][cls_name].append(np.nan)
                    self.metrics_history['cls_recall'][cls_name].append(np.nan)
                    self.metrics_history['cls_map50'][cls_name].append(np.nan)
                    self.metrics_history['cls_map50_95'][cls_name].append(np.nan)

    def save_to_csv(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for metric_name in ['cls_precision', 'cls_recall', 'cls_map50', 'cls_map50_95']:
            df = pd.DataFrame({
                'epoch': self.metrics_history['epoch'],
                **self.metrics_history[metric_name]
            })

            csv_path = os.path.join(save_dir, f'{metric_name}_per_epoch.csv')
            df.to_csv(csv_path, index=False)

        print(f"Метрики по классам сохранены в {save_dir}")


def train_yolo():
    with open('data_for_yolo_1.yaml') as f:
        data = yaml.safe_load(f)
    class_names = data['names']

    model = YOLO('yolo12m.pt')
    metrics_logger = ClassMetricsLogger(class_names)

    def on_val_end(validator):
        metrics_logger.update(validator)

    model.add_callback("on_val_end", on_val_end)

    results = model.train(
        data='data_for_yolo_1.yaml',
        epochs=100,
        batch=16,
        imgsz=544,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        val=True,
        save_json=True,
        plots=True
    )

    metrics_logger.save_to_csv(results.save_dir)

    val_results = model.val()
    print("Обучение завершено.")


if __name__ == "__main__":
    train_yolo()