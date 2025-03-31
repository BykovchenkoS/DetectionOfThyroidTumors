import pandas as pd
import matplotlib.pyplot as plt

csv_file = "retinanet_metrics_node.csv"
data = pd.read_csv(csv_file)

plt.figure(figsize=(15, 12))

# 1. График потерь на обучении
plt.subplot(3, 2, 1)
plt.plot(data['epoch'], data['train_loss'], label='Train Loss', marker='o')
plt.plot(data['epoch'], data['train_box_loss'], label='Box Loss', marker='o')
plt.plot(data['epoch'], data['train_cls_loss'], label='Class Loss', marker='o')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 2. График точности, полноты и F1-меры на валидации
plt.subplot(3, 2, 2)
plt.plot(data['epoch'], data['val_precision'], label='Precision', marker='o')
plt.plot(data['epoch'], data['val_recall'], label='Recall', marker='o')
plt.plot(data['epoch'], data['val_f1'], label='F1 Score', marker='o')
plt.title('Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid()

# 3. График IoU на валидации
plt.subplot(3, 2, 3)
plt.plot(data['epoch'], data['val_iou'], label='IoU', marker='o', color='green')
plt.title('Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.grid()

# 4. График TP, FP, FN на валидации
plt.subplot(3, 2, 4)
plt.plot(data['epoch'], data['tp'], label='True Positives', marker='o')
plt.plot(data['epoch'], data['fp'], label='False Positives', marker='o')
plt.plot(data['epoch'], data['fn'], label='False Negatives', marker='o')
plt.title('Validation TP, FP, FN')
plt.xlabel('Epoch')
plt.ylabel('Count')
plt.legend()
plt.grid()

# 5. График времени обучения
plt.subplot(3, 2, 5)
plt.plot(data['epoch'], data['time'], label='Training Time', marker='o', color='orange')
plt.title('Training Time per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid()

# 6. График скорости обучения (Learning Rate)
plt.subplot(3, 2, 6)
plt.plot(data['epoch'], data['lr'], label='Learning Rate', marker='o', color='purple')
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.legend()
plt.grid()

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Регулируем расстояния
plt.tight_layout(pad=4.0)  # Добавляем общий отступ
output_file = "../predict_retinanet_node/metrics/retinanet_metrics_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')

plt.show()
