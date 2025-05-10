import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = 'predict_yolo_12_node/iou_results_per_object.csv'

data = pd.read_csv(csv_path)
data['image_number'] = data['image_name'].str.extract(r'(\d+)').astype(int)
data = data.sort_values(by='image_number')

iou_values = data['iou']
image_indices = data['image_number']


def plot_and_save(metric_name, metric_values, image_indices, output_dir):
    mean_value = metric_values.mean()

    plt.figure(figsize=(15, 8))
    plt.plot(image_indices, metric_values, marker='o', linestyle='-', color='b',
             label=f'Mean {metric_name} = {mean_value:.4f}')
    plt.title(f'{metric_name} Distribution')
    plt.xlabel('Image index')
    plt.ylabel(metric_name)
    plt.grid(True)
    for i, value in zip(image_indices, metric_values):
        plt.text(i, value, f'{value:.4f}', fontsize=8, ha='center', va='bottom')

    plt.legend(loc='best', fontsize=12)

    output_file = os.path.join(output_dir, f'{metric_name.lower()}_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.show()


output_dir = os.path.dirname(csv_path)
os.makedirs(output_dir, exist_ok=True)

plot_and_save('IoU', iou_values, image_indices, output_dir)
