import os
import random
import shutil
def split_dataset(image_dir, label_dir, dest_dir, test_size=0.2):
    all_images = os.listdir(image_dir)

    test_image_dir = os.path.join(dest_dir, 'val', 'val')
    train_image_dir = os.path.join(dest_dir, 'train', 'val')
    test_label_dir = os.path.join(dest_dir, 'val', 'labels')
    train_label_dir = os.path.join(dest_dir, 'train', 'labels')

    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)

    random.shuffle(all_images)

    num_test = int(len(all_images) * test_size)

    test_images = all_images[:num_test]
    train_images = all_images[num_test:]

    for image in test_images:
        label_file = image.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(label_dir, label_file)):
            shutil.copy(os.path.join(image_dir, image), os.path.join(test_image_dir, image))
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(test_label_dir, label_file))

    for image in train_images:
        label_file = image.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(label_dir, label_file)):
            shutil.copy(os.path.join(image_dir, image), os.path.join(train_image_dir, image))
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))

    print(f"Dataset split completed: {len(train_images)} train for training, {len(test_images)} val for testing.")


image_directory = 'screen foto/dataset 2024-04-21 14_33_36/img'
label_directory = 'screen foto/dataset 2024-04-21 14_33_36/yolo1_ann'
destination_directory = 'screen foto/dataset 2024-04-21 14_33_36/split'

split_dataset(image_directory, label_directory, destination_directory)
