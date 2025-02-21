import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from detr.models.detr import PostProcess, PostProcessSegm
from detr.util.misc import collate_fn as default_collate_fn
from detr.util.misc import NestedTensor
from my_detr.DETR_neural_network import CustomDataset, get_model_instance_segmentation
import json


PREDICTIONS_DIR = "detr_predictions"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


class ValidationDataset(CustomDataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        super().__init__(images_dir, annotations_dir, transforms=transforms)


def get_transform(train=False):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


if __name__ == '__main__':
    val_dataset = ValidationDataset(
        images_dir='dataset_coco_neuro_1/val/images',
        annotations_dir='dataset_coco_neuro_1/val/annotations',
        transforms=get_transform(train=False)
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=default_collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = len(val_dataset.category_map)
    model, _, postprocessors = get_model_instance_segmentation(num_classes)
    model.to(device)

    model_path = "my_detr/detr_screen.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    postprocessor = PostProcess()
    postprocessor_seg = PostProcessSegm()

    image_files = sorted(os.listdir('dataset_coco_neuro_1/val/images'))

    for i, (images, targets) in enumerate(val_data_loader):
        if isinstance(images, NestedTensor):
            tensor_images, masks = images.decompose()
            images = [img.to(device) for img in tensor_images]
        else:
            images = list(image.to(device) for image in images)

        outputs = model(images)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        max_target_sizes = torch.stack([t["size"] for t in targets], dim=0).to(device)
        results = postprocessor(outputs, orig_target_sizes)
        results_seg = postprocessor_seg(results, outputs, orig_target_sizes, max_target_sizes)

        for j, result in enumerate(results_seg):
            image_filename = image_files[i]
            prediction_filename = os.path.splitext(image_filename)[0] + ".json"
            prediction_file = os.path.join(PREDICTIONS_DIR, prediction_filename)

            prediction = {
                "boxes": result["boxes"].cpu().numpy().tolist(),
                "labels": result["labels"].cpu().numpy().tolist(),
                "scores": result["scores"].cpu().numpy().tolist(),
                "masks": result["masks"].cpu().numpy().tolist()
            }

            with open(prediction_file, "w") as f:
                json.dump(prediction, f)

    print(f"Предсказания сохранены в папке: {PREDICTIONS_DIR}")