import os
import random
import shutil

def split_dataset(image_dir, label_dir, dest_dir, test_size=0.2):
    all_images = [f for f in os.listdir(image_dir) if f.startswith("cropped_") and f.endswith(".jpg")]

    test_image_dir = os.path.join(dest_dir, 'images', 'val')
    train_image_dir = os.path.join(dest_dir, 'images', 'train')
    test_label_dir = os.path.join(dest_dir, 'labels', 'val')
    train_label_dir = os.path.join(dest_dir, 'labels', 'train')

    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)

    random.shuffle(all_images)

    num_test = int(len(all_images) * test_size)

    test_images = all_images[:num_test]
    train_images = all_images[num_test:]

    def save_with_new_name(src_image, dest_image_dir, dest_label_dir, label_dir):
        new_image_name = src_image.replace("cropped_", "")
        label_file = new_image_name.replace('.jpg', '.txt')

        if os.path.exists(os.path.join(label_dir, label_file)):
            shutil.copy(os.path.join(image_dir, src_image), os.path.join(dest_image_dir, new_image_name))
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(dest_label_dir, label_file))

    for image in test_images:
        save_with_new_name(image, test_image_dir, test_label_dir, label_dir)

    for image in train_images:
        save_with_new_name(image, train_image_dir, train_label_dir, label_dir)

    print(f"Dataset split completed: {len(train_images)} train for training, {len(test_images)} val for testing.")

image_directory = 'screen foto/dataset 2024-04-21 14_33_36/img_masks_thyroid'
label_directory = 'screen foto/dataset 2024-04-21 14_33_36/yolo2_ann'
destination_directory = 'screen foto/dataset 2024-04-21 14_33_36/split'

split_dataset(image_directory, label_directory, destination_directory)
