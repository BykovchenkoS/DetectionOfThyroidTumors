import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = 'sam_predictions_node_val_metrics/per_object_metrics.csv'
data = pd.read_csv(csv_path)

iou_values = data['iou']
map50_values = data['map50']
map95_values = data['map95']

image_indices = range(len(iou_values))


def plot_and_save(metric_name, metric_values, output_dir):

    mean_value = metric_values.mean()

    plt.figure(figsize=(15, 8))
    plt.plot(image_indices, metric_values, marker='o', linestyle='-', color='b',
             label=f'Mean {metric_name} = {mean_value:.4f}')
    plt.title(f'{metric_name} Distribution')
    plt.xlabel('Image index')
    plt.ylabel(metric_name)
    plt.grid(True)

    for i, value in enumerate(metric_values):
        plt.text(i, value, f'{value:.4f}', fontsize=8, ha='center', va='bottom')

    plt.legend(loc='best', fontsize=12)

    output_file = os.path.join(output_dir, f'{metric_name.lower()}_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.show()


output_dir = os.path.dirname(csv_path)

plot_and_save('IoU', iou_values, output_dir)
plot_and_save('mAP@50', map50_values, output_dir)
plot_and_save('mAP@95', map95_values, output_dir)