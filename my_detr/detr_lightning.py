import logging
import os
import json
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import accuracy_score


log_file = "training_detr.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


def compute_accuracy(preds, targets, iou_threshold=0.5):
    correct_predictions = 0
    total_predictions = 0

    for pred, target in zip(preds, targets):
        pred_boxes = pred["boxes"].cpu().numpy()
        pred_labels = pred["labels"].cpu().numpy()
        target_boxes = target["boxes"].cpu().numpy()
        target_labels = target["labels"].cpu().numpy()

        ious = compute_iou(pred_boxes, target_boxes)
        matched_indices = np.where(ious >= iou_threshold)

        for pred_idx, target_idx in zip(*matched_indices):
            if pred_labels[pred_idx] == target_labels[target_idx]:
                correct_predictions += 1
        total_predictions += len(pred_labels)

    return correct_predictions / total_predictions if total_predictions > 0 else 0.0


def compute_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    inter_x1 = np.maximum(x11, x21.T)
    inter_y1 = np.maximum(y11, y21.T)
    inter_x2 = np.minimum(x12, x22.T)
    inter_y2 = np.minimum(y12, y22.T)

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    union_area = area1 + area2.T - inter_area
    iou = inter_area / union_area
    return iou


def load_mask_and_convert_to_polygons(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            contour = contour.flatten().tolist()
            polygons.append(contour)
    return polygons


def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [polygon], 1)
    return mask


class CustomDataset:
    def __init__(self, images_dir, annotations_dir, processor=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        ann_path = os.path.join(annotations_dir, self.image_files[0].replace('.jpg', '.json'))
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        self.category_map = {category['id']: category['name'] for category in annotation['categories']}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        FIXED_SIZE = (528, 528)

        img_filename = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_filename)
        img = Image.open(img_path).convert("RGB")

        img = img.resize(FIXED_SIZE, Image.Resampling.LANCZOS)

        ann_filename = img_filename.replace('.jpg', '.json')
        ann_path = os.path.join(self.annotations_dir, ann_filename)
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        image_info = annotation["images"][0]
        boxes = []
        labels = []
        masks = []
        segmentation_masks = []

        for ann in annotation["annotations"]:
            xmin, ymin, width, height = ann["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann["category_id"])

            if isinstance(ann["segmentation"], str):
                mask_path = ann["segmentation"]
                if not os.path.exists(mask_path):
                    logging.error(f"Mask file not found: {mask_path}")
                    continue
                polygons = load_mask_and_convert_to_polygons(mask_path)
                segmentation_masks.append(polygons)
                mask = polygons_to_mask(polygons, FIXED_SIZE[0], FIXED_SIZE[1])
                masks.append(mask)
            elif isinstance(ann["segmentation"], list):
                polygons = ann["segmentation"]
                segmentation_masks.append(polygons)
                mask = polygons_to_mask(polygons, FIXED_SIZE[0], FIXED_SIZE[1])
                masks.append(mask)

        coco_annotations = {
            "image_id": image_info["id"],
            "annotations": [
                {
                    "bbox": [xmin, ymin, width, height],
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": width * height,
                    "iscrowd": 0
                }
                for xmin, ymin, width, height, category_id, segmentation in zip(
                    [b[0] for b in boxes],
                    [b[1] for b in boxes],
                    [b[2] - b[0] for b in boxes],
                    [b[3] - b[1] for b in boxes],
                    labels,
                    segmentation_masks
                )
            ],
        }

        if self.processor:
            encoding = self.processor(
                images=img,
                annotations=coco_annotations,
                return_tensors="pt",
                return_segmentation_masks=True
            )
            pixel_values = encoding["pixel_values"].squeeze()
            if "labels" in encoding and len(encoding["labels"]) > 0:
                label_data = encoding["labels"][0]
                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "class_labels": torch.tensor(labels, dtype=torch.int64),
                    "masks": torch.tensor(np.array(masks), dtype=torch.bool),
                }
            else:
                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "class_labels": torch.tensor(labels, dtype=torch.int64),
                    "masks": torch.tensor(np.array(masks), dtype=torch.bool),
                }
        else:
            pixel_values = img
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "class_labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.tensor(np.array(masks), dtype=torch.bool),
            }

        # logging.info(f"Final boxes: {target['boxes']}")

        return pixel_values, target, coco_annotations


def visualize_sample(image, target, title="Sample"):
    fig, ax = plt.subplots(1)

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    ax.imshow(image)

    boxes = target["boxes"]
    labels = target["class_labels"]

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"Class {label}", color='white', fontsize=10, backgroundcolor='red')

    masks = target["masks"]
    if isinstance(masks, torch.Tensor):
        masks = masks.numpy()

    for i, mask in enumerate(masks):
        mask = mask.astype(np.float32)
        mask = np.ma.masked_where(mask == 0, mask)
        ax.imshow(mask, alpha=0.5, cmap='viridis')

    plt.title(title)
    plt.show()


class DetrLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_labels=2):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), prog_bar=True)

        # if batch_idx == 0:
        #     for i in range(len(batch["pixel_values"])):
        #         visualize_sample(
        #             batch["pixel_values"][i],
        #             batch["labels"][i],
        #             title=f"Training Sample {i}"
        #         )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v.item(), prog_bar=True)

        # if batch_idx == 0:
        #     for i in range(len(batch["pixel_values"])):
        #         visualize_sample(batch["pixel_values"][i], batch["labels"][i], title=f"Validation Sample {i}")

        outputs = self.model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
        preds = [
            {
                "boxes": outputs.pred_boxes[i],
                "scores": outputs.logits[i].softmax(-1).max(-1)[0],
                "labels": outputs.logits[i].softmax(-1).argmax(-1)
            }
            for i in range(len(outputs.pred_boxes))
        ]
        targets = [
            {"boxes": t["boxes"], "labels": t["class_labels"]}
            for t in batch["labels"]
        ]
        self.val_map.update(preds, targets)

        accuracy = compute_accuracy(preds, targets)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        map_metrics = self.val_map.compute()
        self.log("val_mAP", map_metrics["map"], prog_bar=True)
        self.log("val_mAP_50", map_metrics["map_50"], prog_bar=True)
        self.val_map.reset()

        logging.info(
            f"Epoch {self.current_epoch} Metrics: "
            f"Loss={self.trainer.callback_metrics['val_loss']:.4f}, "
            f"mAP={map_metrics['map']:.4f}, "
            f"mAP_50={map_metrics['map_50']:.4f}, "
            f"Accuracy={self.trainer.callback_metrics['val_accuracy']:.4f}"
        )

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone},
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    pixel_masks = [torch.ones((item[0].shape[-2], item[0].shape[-1]), dtype=torch.bool) for item in batch]

    pixel_values = torch.stack(pixel_values).float()
    pixel_masks = torch.stack(pixel_masks).bool()

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_masks,
        "labels": targets,
    }


if __name__ == "__main__":
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        do_normalize=False,
        do_resize=False
    )

    train_dataset = CustomDataset(
        images_dir="../dataset_coco_neuro_1/train/images",
        annotations_dir="../dataset_coco_neuro_1/train/annotations",
        processor=processor
    )

    for i in range(min(3, len(train_dataset))):
        img, target, coco_annotations = train_dataset[i]

        # logging.info(f"Sample {i}:")
        # logging.info(f"Image shape: {img.shape}")
        # logging.info(f"Target keys: {target.keys()}")
        # logging.info(f"Boxes shape: {target['boxes'].shape}")
        # logging.info(f"Labels shape: {target['class_labels'].shape}")
        # if "masks" in target:
        #     logging.info(f"Masks shape: {target['masks'].shape}")

    val_dataset = CustomDataset(
        images_dir="../dataset_coco_neuro_1/val/images",
        annotations_dir="../dataset_coco_neuro_1/val/annotations",
        processor=processor
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = DetrLightningModule(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_labels=len(train_dataset.category_map))

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    model_save_path = "detr_model.pth"
    torch.save(model.model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")