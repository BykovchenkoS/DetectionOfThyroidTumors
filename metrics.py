import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из CSV файла
df = pd.read_csv('mask_rcnn/maskRCNN_result.csv')  # Замените 'your_file.csv' на путь к вашему файлу

# Выводим первые несколько строк, чтобы убедиться, что данные загружены корректно
print(df.head())

# Построение графиков

# График потерь
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train/box_loss'], label='Train Box Loss', marker='o')
plt.plot(df['Epoch'], df['Train/cls_loss'], label='Train Class Loss', marker='o')
plt.plot(df['Epoch'], df['Train/dfl_loss'], label='Train DFL Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Losses')
plt.legend()
plt.grid(True)
plt.show()

# График метрик (Precision, Recall, mAP50, mAP50-95)
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Metrics/precision(B)'], label='Precision', marker='o')
plt.plot(df['Epoch'], df['Metrics/recall(B)'], label='Recall', marker='o')
plt.plot(df['Epoch'], df['Metrics/mAP50(B)'], label='mAP50', marker='o')
plt.plot(df['Epoch'], df['Metrics/mAP50-95(B)'], label='mAP50-95', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Metrics')
plt.legend()
plt.grid(True)
plt.show()

# График потерь на валидации (если данные есть)
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Val/box_loss'], label='Val Box Loss', marker='o')
plt.plot(df['Epoch'], df['Val/cls_loss'], label='Val Class Loss', marker='o')
plt.plot(df['Epoch'], df['Val/dfl_loss'], label='Val DFL Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Losses')
plt.legend()
plt.grid(True)
plt.show()

# График изменения Learning Rate
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['LR/pg0'], label='Learning Rate', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate')
plt.legend()
plt.grid(True)
plt.show()
