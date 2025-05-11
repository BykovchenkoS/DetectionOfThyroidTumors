import os
import pandas as pd
import matplotlib.pyplot as plt


def load_iou_data(csv_path):
    data = pd.read_csv(csv_path)

    iou_data = (
        data.groupby(["image_name", "predicted_class"])["iou"]
        .mean()
        .reset_index()
    )

    iou_data["image_number"] = iou_data["image_name"].str.extract(r"(\d+)").astype(int)
    iou_data = iou_data.sort_values(by="image_number")

    overall_iou_data = (
        data.groupby("image_name")["iou"].mean().reset_index()
    )
    overall_iou_data["image_number"] = overall_iou_data["image_name"].str.extract(r"(\d+)").astype(int)
    overall_iou_data = overall_iou_data.sort_values(by="image_number")
    overall_mean_iou = data["iou"].mean()

    return iou_data, overall_iou_data, overall_mean_iou


def plot_iou_graph(iou_data, overall_iou_data, overall_mean_iou, output_dir):
    class_0_data = iou_data[iou_data["predicted_class"] == "Thyroid tissue"]
    class_1_data = iou_data[iou_data["predicted_class"] == "Carotis"]

    plt.figure(figsize=(14, 7))
    plt.plot(
        class_0_data["image_number"],
        class_0_data["iou"],
        marker='^',
        linestyle=':',
        color='green',
        label='Thyroid tissue'
    )

    plt.plot(
        class_1_data["image_number"],
        class_1_data["iou"],
        marker='s',
        linestyle='--',
        color='orange',
        label='Carotis'
    )

    plt.plot(
        overall_iou_data["image_number"],
        overall_iou_data["iou"],
        marker='o',
        linestyle='-',
        color='b',
        label='Mean IoU per Image'
    )

    plt.plot([], [], ' ', label=f'Overall mean IoU = {overall_mean_iou:.4f}')

    plt.title('IoU Distribution')
    plt.xlabel("Image index")
    plt.ylabel("IoU")
    plt.grid(True)

    max_index = max(iou_data["image_number"])
    plt.xticks(range(0, max_index + 1, 50))
    plt.legend()
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'IoU_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    csv_path = 'predict_yolo_12_screen/iou_results_per_object.csv'

    output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    iou_data, overall_iou_data, overall_mean_iou = load_iou_data(csv_path)
    plot_iou_graph(iou_data, overall_iou_data, overall_mean_iou, output_dir)


if __name__ == "__main__":
    main()
