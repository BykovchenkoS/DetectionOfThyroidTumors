import os
import pandas as pd
import matplotlib.pyplot as plt


def load_iou_data(csv_path):
    data = pd.read_csv(csv_path)

    iou_data = data.groupby(["image_name", "class_id"])["iou"].mean().reset_index()
    iou_data["image_number"] = iou_data["image_name"].str.extract(r"(\d+)").astype(int)
    iou_data = iou_data.sort_values(by="image_number")

    overall_iou_data = data.groupby("image_name")["iou"].mean().reset_index()
    overall_iou_data["image_number"] = overall_iou_data["image_name"].str.extract(r"(\d+)").astype(int)
    overall_iou_data = overall_iou_data.sort_values(by="image_number")

    return iou_data, overall_iou_data, data["iou"].mean()


def filter_map_data(map_data):
    filtered = map_data[(map_data["map50_Carotis"] != 0) & (map_data["map95_Carotis"] != 0)]
    filtered = map_data[
        (map_data["map50_Carotis"] != 0) & (map_data["map95_Carotis"] != 0)
        ].copy()

    filtered.loc[:, "image_number"] = (
        filtered["image_name"]
        .str.extract(r"(\d+)", expand=False)
        .fillna(0)
        .astype(int)
    )

    filtered = filtered.sort_values(by="image_number")
    return filtered


def plot_iou_graph(iou_data, overall_iou_data, overall_mean_iou, output_dir):
    class_0_data = iou_data[iou_data["class_id"] == 0]
    class_1_data = iou_data[iou_data["class_id"] == 1]

    plt.figure(figsize=(14, 7))

    plt.plot(class_0_data["image_number"], class_0_data["iou"],
             marker='^', linestyle=':', color='green', label='Thyroid tissue')
    plt.plot(class_1_data["image_number"], class_1_data["iou"],
             marker='s', linestyle='--', color='orange', label='Carotis')
    plt.plot(overall_iou_data["image_number"], overall_iou_data["iou"],
             marker='o', linestyle='-', color='b', label='Mean IoU per Image')

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


def plot_map_graph(map_data, output_dir):
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    axs[0].plot(
        map_data["image_number"], map_data["map50_Carotis"],
        marker='^', linestyle='-', color='red', label='mAP50'
    )
    axs[0].plot(
        map_data["image_number"], map_data["map95_Carotis"],
        marker='o', linestyle='--', color='blue', label='mAP95'
    )

    axs[0].set_title('mAP Distribution for Carotis')
    axs[0].set_xlabel('Image index')
    axs[0].set_ylabel('mAP')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_xticks(range(0, max(map_data["image_number"]) + 1, 50))

    axs[1].plot(
        map_data["image_number"], map_data["map50_Thyroid tissue"],
        marker='^', linestyle='-', color='orange', label='mAP50'
    )
    axs[1].plot(
        map_data["image_number"], map_data["map95_Thyroid tissue"],
        marker='o', linestyle='--', color='green', label='mAP95'
    )

    axs[1].set_title('mAP Distribution for Thyroid tissue')
    axs[1].set_xlabel('Image index')
    axs[1].set_ylabel('mAP')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_xticks(range(0, max(map_data["image_number"]) + 1, 50))

    plt.tight_layout()

    map_output_file = os.path.join(output_dir, 'mAP_distribution.png')
    plt.savefig(map_output_file, dpi=300, bbox_inches='tight')

    plt.show()


def main():
    csv_path = 'sam_predictions_screen_val_metrics/per_object_metrics.csv'
    map_csv_path = 'sam_predictions_screen_val_metrics/image_map.csv'

    output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    iou_data, overall_iou_data, overall_mean_iou = load_iou_data(csv_path)
    plot_iou_graph(iou_data, overall_iou_data, overall_mean_iou, output_dir)

    map_data = pd.read_csv(map_csv_path)
    filtered_map_data = filter_map_data(map_data)
    plot_map_graph(filtered_map_data, output_dir)


if __name__ == "__main__":
    main()
