import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('detect/detect/train8/results.csv')
epochs = data['epoch']
map95 = data['metrics/mAP50-95(B)']

plt.figure(figsize=(10, 6))

plt.plot(epochs, map95, label='Node', color='purple', linestyle='-', marker='o')

plt.title('mAP95 metric evolution by epoch', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('mAP95', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

plt.savefig('detect/detect/train8/mAP95_curve.png', dpi=300)

plt.tight_layout()
plt.show()
