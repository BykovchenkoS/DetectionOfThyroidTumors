import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def save_image_with_predictions(image, predictions, class_names, output_path, threshold=0.5):
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    labels = predictions[0]['labels'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()
    masks = predictions[0]['masks'].cpu().detach().numpy()

    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    masks = masks[keep]

    color_map = {
        'sagital_longitudinal': 'blue',
        'Thyroid tissue': 'red',
        'Carotis': 'green',
        'background': 'yellow'
    }

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    for box, label, mask in zip(boxes, labels, masks):
        class_name = class_names[label]
        color = color_map[class_name]

        ax.add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor=color, linewidth=2))
        ax.text(box[0], box[1] - 10, class_name, color=color, fontsize=12, backgroundcolor='white')

        mask = mask[0]
        image_with_mask = np.array(image).astype(np.float32)

        if class_name == 'sagital_longitudinal':
            image_with_mask[mask > 0.5] = [0, 0, 255]  # синий для 'sagital_longitudinal'
        elif class_name == 'Thyroid tissue':
            image_with_mask[mask > 0.5] = [255, 0, 0]  # красный для 'Thyroid tissue'
        elif class_name == 'Carotis':
            image_with_mask[mask > 0.5] = [0, 255, 0]  # зеленый для 'Carotis'
        elif class_name == 'background':
            image_with_mask[mask > 0.5] = [255, 255, 0]  # желтый для 'background'

        image_with_mask /= 255.0
        ax.imshow(image_with_mask, alpha=0.5)

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


class_names = ['sagital_longitudinal', 'Thyroid tissue', 'Carotis', 'background']

model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = len(class_names)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

model.load_state_dict(torch.load('mask_rcnn_model_1.pth', weights_only=True))
model.eval()

transform = T.Compose([T.ToTensor()])
input_folder = 'screen foto/dataset 2024-04-21 14_33_36/img/'
output_folder = 'predict_mask_rcnn_screen/'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        image_tensor = image_tensor.to(device)
        model.to(device)

        # выполнение предсказания
        with torch.no_grad():
            prediction = model(image_tensor)

        output_path = os.path.join(output_folder, f"pred_{filename}")
        save_image_with_predictions(image, prediction, class_names, output_path, threshold=0.5)

print("Предсказания сохранены.")
