import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('detect/detect/train7/cls_map50_95_per_epoch.csv')
epochs = data['epoch']
thyroid_tissue = data['Thyroid tissue']
carotis = data['Carotis']

average_map95 = (thyroid_tissue + carotis) / 2

plt.figure(figsize=(10, 6))

plt.plot(epochs, thyroid_tissue, label='Thyroid tissue', color='blue', linestyle='-', marker='o')

plt.plot(epochs, carotis, label='Carotis', color='orange', linestyle='--', marker='s')

plt.plot(epochs, average_map95, label='Average mAP95', color='green', linestyle='-.', marker='^')

plt.title('mAP95 metric evolution by epoch', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('mAP95', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

plt.savefig('detect/detect/train7/mAP95_curve.png', dpi=300)

plt.tight_layout()
