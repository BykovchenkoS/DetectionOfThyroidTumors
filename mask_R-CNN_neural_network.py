import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from tqdm import tqdm
from coco_for_pytorch import CustomDataset
from coco_for_pytorch import category_map
import csv
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("training_node.log"),
    logging.StreamHandler()
])

model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

num_classes = len(category_map) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256,
                                                                                          num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

transform = T.Compose([T.ToTensor()])

dataset = CustomDataset(images_dir='dataset_coco_neuro_2/train/images',
                        annotations_dir='dataset_coco_neuro_2/train/annotations',
                        transforms=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

with open('maskRCNN_result_node.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time', 'Train/box_loss', 'Train/cls_loss', 'Train/dfl_loss',
                     'Metrics/precision(B)', 'Metrics/recall(B)', 'Metrics/mAP50(B)', 'Metrics/mAP50-95(B)',
                     'Val/box_loss', 'Val/cls_loss', 'Val/dfl_loss', 'LR/pg0'])

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        for images, targets, _ in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            box_loss = loss_dict['loss_box_reg'].item()
            cls_loss = loss_dict['loss_classifier'].item()
            dfl_loss = loss_dict['loss_objectness'].item()

            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            losses.backward()
            optimizer.step()

        epoch_time = time.time() - start_time

        writer.writerow([epoch + 1, epoch_time, box_loss, cls_loss, dfl_loss,
                         'precision', 'recall', 'mAP50', 'mAP50-95',
                         'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
                         optimizer.param_groups[0]['lr']])
        file.flush()

        log_message = (f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(data_loader)} - "
                       f"Box Loss: {box_loss} - Class Loss: {cls_loss} - Objectness Loss: {dfl_loss}")
        logging.info(log_message)
        print(log_message)

torch.save(model.state_dict(), 'mask_rcnn_model_node.pth')
