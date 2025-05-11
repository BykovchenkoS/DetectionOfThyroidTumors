import os
import json
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry
from tqdm import tqdm

CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
FINE_TUNED_MODEL_PATH = "sam_best_node.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INFERENCE_IMAGES_DIR = "dataset_sam_neuro_2/val/images"
ANNOTATIONS_DIR = "dataset_sam_neuro_2/val/annotations"
OUTPUT_DIR = "sam_predictions_visualized_node"
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_annotations(annotations_dir, image_name):
    base_name = os.path.splitext(image_name)[0]
    annotation_path = os.path.join(annotations_dir, f"{base_name}.json")

    if not os.path.exists(annotation_path):
        return None

    with open(annotation_path) as f:
        data = json.load(f)

    image_id = None
    for img_info in data['images']:
        if img_info['file_name'].endswith(image_name):
            image_id = img_info['id']
            break

    if image_id is None:
        return None

    bboxes = []
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            bboxes.append(ann['bbox'])

    return bboxes


def convert_bbox(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def save_results(image, mask, output_path, mask_path):
    """Сохраняет визуализацию и маску"""
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    visualization = image.copy()
    cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)

    color_mask = np.zeros_like(image)
    color_mask[..., 0] = 255
    masked = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
    visualization[mask == 1] = masked[mask == 1]

    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(DEVICE)

# Загрузка дообученных весов
checkpoint = torch.load(FINE_TUNED_MODEL_PATH, map_location=DEVICE)
sam.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
sam.eval()

image_files = [f for f in os.listdir(INFERENCE_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

for img_name in tqdm(image_files, desc="Inference"):
    try:
        img_path = os.path.join(INFERENCE_IMAGES_DIR, img_name)
        original_image = load_image(img_path)
        height, width = original_image.shape[:2]

        bboxes = load_annotations(ANNOTATIONS_DIR, img_name)
        box_torch = None

        if bboxes and len(bboxes) > 0:
            print(f"\nOriginal bbox: {bboxes[0]}")
            bbox = convert_bbox(bboxes[0])
            print(f"Converted bbox (x1,y1,x2,y2): {bbox}")

            if bbox[2] - bbox[0] < 10 or bbox[3] - bbox[1] < 10:
                print("Внимание: bbox слишком маленький, используется prompt-free режим")
            else:
                box_torch = torch.as_tensor(bbox, dtype=torch.float, device=DEVICE)
                box_torch = box_torch[None, :]

        image_tensor = torch.tensor(original_image).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_embeddings = sam.image_encoder(image_tensor)

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

        mask = torch.nn.functional.interpolate(
            low_res_masks,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        binary_mask = (mask > 0.3).astype(np.uint8)

        base_name = os.path.splitext(img_name)[0]
        save_results(
            original_image,
            binary_mask,
            os.path.join(OUTPUT_DIR, f"vis_{base_name}.png"),
            os.path.join(MASKS_DIR, f"mask_{base_name}.png")
        )

        print(f"\n{img_name}: Использован bbox: {bboxes[0] if bboxes else 'Нет'}, "
              f"IOU={iou_predictions.item():.3f}, Площадь={np.sum(binary_mask)} пикселей")

    except Exception as e:
        print(f"\nОшибка при обработке {img_name}: {str(e)}")
        continue

print(f"\nРезультаты сохранены в:\n- Визуализации: {OUTPUT_DIR}\n- Маски: {MASKS_DIR}")
