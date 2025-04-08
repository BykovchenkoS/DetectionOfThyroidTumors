from PIL import Image
import os


def add_padding_to_mask(mask_path, output_path, target_size):
    mask = Image.open(mask_path).convert("L")
    original_width, original_height = mask.size

    pad_left = (target_size[0] - original_width) // 2
    pad_right = target_size[0] - original_width - pad_left
    pad_top = (target_size[1] - original_height) // 2
    pad_bottom = target_size[1] - original_height - pad_top

    padded_mask = Image.new("L", target_size, color=0)
    padded_mask.paste(mask, (pad_left, pad_top))

    padded_mask.save(output_path)


def process_all_masks(masks_dir, output_masks_dir, target_size):
    # Создаем выходную директорию, если её нет
    os.makedirs(output_masks_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for mask_file in mask_files:
        mask_path = os.path.join(masks_dir, mask_file)
        output_mask_path = os.path.join(output_masks_dir, mask_file)
        add_padding_to_mask(mask_path, output_mask_path, target_size)

        print(f"Processed {mask_file} and saved to {output_mask_path}")


masks_dir = 'dataset_sam_neuro_2/masks/'
output_masks_dir = 'dataset_sam_neuro_2/processed_masks/'
target_size = (1024, 1024)

process_all_masks(masks_dir, output_masks_dir, target_size)
