import os
import shutil


def copy_matching_files(folder1, folder2, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files_in_folder1 = set(os.listdir(folder1))
    files_in_folder2 = os.listdir(folder2)

    # Проходим по файлам второй папки и проверяем их наличие в первой папке
    for file in files_in_folder2:
        if file in files_in_folder1:
            source_path = os.path.join(folder2, file)
            destination_path = os.path.join(output_folder, file)

            shutil.copy2(source_path, destination_path)
            print(f"Скопирован файл: {file}")


folder1 = "dataset_coco_neuro_2/train/images"
folder2 = "dataset_coco_neuro_3/images_neuro_3"
output_folder = "dataset_coco_neuro_3/train/images"

copy_matching_files(folder1, folder2, output_folder)
