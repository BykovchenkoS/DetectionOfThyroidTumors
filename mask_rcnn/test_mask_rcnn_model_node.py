import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize
import json
import skimage.draw
from scipy.ndimage import zoom


def save_image_with_predictions(image, predictions, class_names, output_path,
                                threshold=0.5, ground_truth=None, true_mask=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 528 / 100, 528 / 100))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1)

    plot_predictions(ax1, image, predictions, class_names, threshold, title="Predictions")
    plot_ground_truth(ax2, image, ground_truth, class_names, title="Ground Truth", true_mask=true_mask)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=100)
    plt.close()


def plot_predictions(ax, image, predictions, class_names, threshold, title):
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    labels = predictions[0]['labels'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()
    masks = predictions[0]['masks'].cpu().detach().numpy()

    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    masks = masks[keep]

    color_map = {
        'Node': 'purple',
        'background': 'yellow'
    }

    ax.imshow(image)
    ax.set_title(title)

    for box, label, mask in zip(boxes, labels, masks):
        class_name = class_names[label]
        color = color_map.get(class_name, 'black')

        ax.add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                          fill=False, edgecolor=color, linewidth=2)
        )

        ax.text(box[0], box[1] - 10, class_name, color=color,
                fontsize=12, backgroundcolor='white')

        mask = mask[0]
        h, w = image.size[1], image.size[0]

        x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            box_mask = mask > 0.5
            box_mask_resized = zoom(
                box_mask,
                (float(y2 - y1) / box_mask.shape[0], float(x2 - x1) / box_mask.shape[1]),
                order=0
            )

            full_mask = np.zeros((h, w))
            full_mask[y1:y2, x1:x2] = box_mask_resized

            image_with_mask = np.array(image).astype(np.float32) / 255.0
            if class_name == 'Node':
                image_with_mask[full_mask > 0.5] = [0.5, 0, 0.5]  # purple
            elif class_name == 'background':
                image_with_mask[full_mask > 0.5] = [1, 1, 0]  # yellow

            ax.imshow(image_with_mask, alpha=0.5)

    ax.axis('off')


def plot_ground_truth(ax, image, ground_truth, class_names, title, true_mask=None):
    ax.imshow(image)
    ax.set_title(title)

    color_map = {
        'Node': 'green',
        'background': 'blue'
    }

    for ann in ground_truth:
        box = ann['bbox']
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

        class_id = ann['category_id']
        class_name = class_names[class_id]
        color = color_map.get(class_name, 'black')

        ax.add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                          fill=False, edgecolor=color, linewidth=2))
        ax.text(box[0], box[1] - 10, class_name, color=color,
                fontsize=12, backgroundcolor='white')

    if true_mask is not None:
        image_with_mask = np.array(image).astype(np.float32) / 255.0
        image_with_mask[true_mask > 0] = [0, 0.5, 0]
        ax.imshow(image_with_mask, alpha=0.3)

    ax.axis('off')


def load_ground_truth(image_filename, annotations_folder):
    base_name = os.path.splitext(image_filename)[0]
    annotation_path = os.path.join(annotations_folder, f"{base_name}.json")

    if not os.path.exists(annotation_path):
        return None

    with open(annotation_path) as f:
        data = json.load(f)

    return data.get('annotations', [])


def load_true_mask(image_filename, masks_folder):
    base_name = os.path.splitext(image_filename)[0]
    base_name = base_name.split('_')[0]

    matching_masks = [f for f in os.listdir(masks_folder)
                      if f.startswith(f"{base_name}_") and f.endswith('.png')]

    if not matching_masks:
        return None

    mask_path = os.path.join(masks_folder, matching_masks[0])

    try:
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)

        mask_array = (mask_array > 128).astype(np.float32)
        return mask_array

    except Exception as e:
        print(f"Ошибка загрузки маски: {e}")
        return None


class_names = ['background', 'Node']
model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
num_classes = len(class_names)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

model.load_state_dict(torch.load('mask_rcnn_model_node.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
    min_size=528, max_size=528, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
)

transform = T.Compose([T.ToTensor()])

input_folder = '../dataset_coco_neuro_3/images_neuro_3/'
annotations_folder = '../dataset_coco_neuro_3/ann/'
masks_folder = '../dataset_coco_neuro_3/masks/'

output_folder = '../predict_mask_rcnn_node/'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("RGB")
        ground_truth = load_ground_truth(filename, annotations_folder)
        true_mask = load_true_mask(filename, masks_folder)
        image_tensor = transform(image).unsqueeze(0)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        image_tensor = image_tensor.to(device)
        model.to(device)

        with torch.no_grad():
            prediction = model(image_tensor)
        output_path = os.path.join(output_folder, f"pred_{filename}")
        save_image_with_predictions(image, prediction, class_names, output_path, threshold=0.5,
                                    ground_truth=ground_truth, true_mask=true_mask)

print("Предсказания сохранены.")
