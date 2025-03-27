import os
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'retinanet_model_node.pth'
    test_image_dir = '../dataset_coco_neuro_3/test_img'
    test_annotations_dir = '../dataset_coco_neuro_3/test_ann'
    output_dir = '../predict_retinanet_node'
    confidence_threshold = 0.5
    class_names = {
        1: 'Node',
        2: 'Background'
    }
    colors = {
        'prediction': 'red',
        'ground_truth': 'green'
    }


def load_model():
    model = retinanet_resnet50_fpn(num_classes=len(Config.class_names))
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.eval()
    return model.to(Config.device)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(Config.device)
    return image, image_tensor.unsqueeze(0)


def load_annotations(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(Config.test_annotations_dir, f"{base_name}.json")

    if not os.path.exists(annotation_path):
        return None

    with open(annotation_path) as f:
        data = json.load(f)

    return data.get('annotations', [])

def visualize_predictions(image, predictions, ground_truth=None):
    fig, ax = plt.subplots(1, 2 if ground_truth else 1, figsize=(20, 10))

    if ground_truth:
        ax[0].imshow(image)
        plot_boxes(ax[0], predictions, Config.colors['prediction'], 'Predictions')

        ax[1].imshow(image)
        plot_boxes(ax[1], ground_truth, Config.colors['ground_truth'], 'Ground Truth')
    else:
        ax.imshow(image)
        plot_boxes(ax, predictions, Config.colors['prediction'], 'Predictions')

    plt.tight_layout()
    return fig


def plot_boxes(ax, boxes_data, color, title):
    ax.set_title(title)

    if isinstance(boxes_data, dict):
        boxes = boxes_data['boxes'].cpu().numpy()
        labels = boxes_data['labels'].cpu().numpy()
        scores = boxes_data['scores'].cpu().numpy()

        keep = scores >= Config.confidence_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        for box, label, score in zip(boxes, labels, scores):
            draw_box(ax, box, Config.class_names.get(label, str(label)),
                     color, f"{score:.2f}")
    else:
        for ann in boxes_data:
            box = ann['bbox']
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            label_id = ann['category_id']
            draw_box(ax, box, Config.class_names.get(label_id, str(label_id)), color)

    ax.axis('off')


def draw_box(ax, box, label, color, score=None):
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin

    rect = plt.Rectangle(
        (xmin, ymin), width, height,
        fill=False, edgecolor=color, linewidth=2
    )
    ax.add_patch(rect)

    text = f"{label}: {score}" if score else label
    ax.text(
        xmin, ymin - 5, text,
        bbox=dict(facecolor=color, alpha=0.5),
        fontsize=8, color='white'
    )


def main():
    model = load_model()
    os.makedirs(Config.output_dir, exist_ok=True)

    for image_file in tqdm(os.listdir(Config.test_image_dir)):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(Config.test_image_dir, image_file)

        try:
            image, image_tensor = load_image(image_path)
            ground_truth = load_annotations(image_path)

            with torch.no_grad():
                predictions = model(image_tensor)[0]

            fig = visualize_predictions(image, predictions, ground_truth)

            output_path = os.path.join(Config.output_dir, f"pred_{image_file}")
            fig.savefig(output_path, bbox_inches='tight', dpi=100)
            plt.close(fig)

        except Exception as e:
            print(f"Ошибка при обработке {image_file}: {str(e)}")


if __name__ == "__main__":
    main()
    print("Визуализация предсказаний завершена.")
