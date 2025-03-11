import logging
import os
import json
import sys
import torch
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision


log_file = "training_detr_lightning_screen.log"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)


class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, processor=None, transforms=None, masks_output_dir=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.processor = processor
        self.transforms = transforms
        self.masks_output_dir = masks_output_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

        ann_path = os.path.join(annotations_dir, self.image_files[0].replace('.jpg', '.json'))
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        self.category_map = {category['id']: category['name'] for category in annotation['categories']}

        if masks_output_dir and not os.path.exists(masks_output_dir):
            os.makedirs(masks_output_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_filename = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_filename)
        img = Image.open(img_path).convert("RGB")

        ann_filename = img_filename.replace('.jpg', '.json')
        ann_path = os.path.join(self.annotations_dir, ann_filename)
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        image_info = annotation["images"][0]
        img_width, img_height = image_info["width"], image_info["height"]

        boxes = []
        labels = []
        masks = []

        for ann in annotation["annotations"]:
            xmin, ymin, width, height = ann["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann["category_id"])

            if isinstance(ann["segmentation"], str):
                mask_path = ann["segmentation"]
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask)
                mask = (mask > 0).astype(np.uint8)
                masks.append(mask)
            elif isinstance(ann["segmentation"], list):
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                segmentation = np.array(ann["segmentation"])
                polygon = segmentation.reshape(-1, 2).astype(np.int32)
                mask = cv2.fillPoly(mask, [polygon], 1)
                masks.append(mask)

        coco_annotations = {
            "image_id": image_info["id"],
            "annotations": [
                {
                    "bbox": [xmin, ymin, width, height],
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": ann.get("area", width * height),
                    "iscrowd": ann.get("iscrowd", 0)
                }
                for xmin, ymin, width, height, category_id, segmentation, ann in zip(
                    [b[0] for b in boxes],
                    [b[1] for b in boxes],
                    [b[2] - b[0] for b in boxes],
                    [b[3] - b[1] for b in boxes],
                    labels,
                    [ann["segmentation"] for ann in annotation["annotations"]],
                    annotation["annotations"],
                )
            ],
        }

        # Преобразование данных с помощью DetrImageProcessor
        if self.processor:
            encoding = self.processor(
                images=img,
                annotations=coco_annotations,
                return_tensors="pt",
            )
            pixel_values = encoding["pixel_values"].squeeze()

            if "labels" in encoding and len(encoding["labels"]) > 0:
                label_data = encoding["labels"][0]
                target = {
                    "boxes": label_data.get("boxes", torch.tensor([])),
                    "class_labels": label_data.get("class_labels", torch.tensor([])),
                    "masks": label_data.get("masks", torch.tensor([])),
                }
            else:
                target = {
                    "boxes": torch.tensor([]),
                    "class_labels": torch.tensor([]),
                    "masks": torch.tensor([]),
                }
        else:
            pixel_values = img
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "class_labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.tensor(masks, dtype=torch.uint8),
            }

        return pixel_values, target


class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_labels):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path="facebook/detr-resnet-50",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        # Добавляем метрику mAP
        self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]

        labels = [
            {
                "boxes": t["boxes"].to(self.device),
                "class_labels": t["class_labels"].to(self.device),
                "masks": t["masks"].to(self.device) if "masks" in t else None,
            }
            for t in batch["labels"]
        ]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)

        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item())

        if batch_idx % 10 == 0:
            print(f"Training Loss at step {batch_idx}: {loss.item():.4f}")

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        targets = batch["labels"]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)

        pred_logits = outputs.logits
        pred_boxes = outputs.pred_boxes


        preds = [
            {
                "boxes": pred_boxes[i],
                "scores": pred_logits[i].softmax(-1).max(-1)[0],
                "labels": pred_logits[i].softmax(-1).argmax(-1)
            }
            for i in range(len(pred_boxes))
        ]

        targets_for_metric = [
            {
                "boxes": t["boxes"],
                "labels": t["class_labels"]
            }
            for t in targets
        ]

        self.val_map.update(preds, targets_for_metric)

        return loss

    def on_validation_epoch_end(self):
        map_dict = self.val_map.compute()

        self.log("validation/mAP", map_dict["map"])
        self.log("validation/mAP_50", map_dict["map_50"])

        self.val_map.reset()

        avg_loss = self.trainer.logged_metrics.get("validation_loss", torch.tensor(0.0)).item()

        logging.info(f"Validation Results - Epoch {self.current_epoch}:")
        logging.info(f"Loss: {avg_loss:.4f}")
        logging.info(f"mAP: {map_dict['map'].item():.4f}")
        logging.info(f"mAP_50: {map_dict['map_50'].item():.4f}")

        print(f"\nValidation Results - Epoch {self.current_epoch}:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"mAP: {map_dict['map'].item():.4f}")
        print(f"mAP_50: {map_dict['map_50'].item():.4f}\n")

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    pixel_masks = [torch.ones((item[0].shape[-2], item[0].shape[-1]), dtype=torch.bool) for item in batch]

    pixel_values = torch.stack(pixel_values)
    pixel_masks = torch.stack(pixel_masks)

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_masks,
        "labels": targets,
    }


if __name__ == '__main__':
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = CustomDataset(
        images_dir="dataset_coco_neuro_1/train/images",
        annotations_dir="dataset_coco_neuro_1/train/annotations",
        processor=processor
    )

    val_dataset = CustomDataset(
        images_dir="dataset_coco_neuro_1/val/images",
        annotations_dir="dataset_coco_neuro_1/val/annotations",
        processor=processor
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = Detr(
        lr=1e-5,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        num_labels=len(train_dataset.category_map)
    )

    model.train_loader = train_loader
    model.val_loader = val_loader

    trainer = pl.Trainer(
        devices=1,
        accelerator="cpu",
        max_epochs=100,
        gradient_clip_val=0.1,
        accumulate_grad_batches=8,
        log_every_n_steps=5
    )

    trainer.fit(model)

    model_save_path = "detr_model_screen_v2.pth"
    torch.save(model.model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")