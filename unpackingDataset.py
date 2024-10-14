import os
import tarfile


def safe_extract(tar, path=".", name_map=None):
    if name_map is None:
        name_map = {}

    for member in tar.getmembers():
        member.name = member.name.replace(":", "_")
        if member.name in name_map:
            member.name = name_map[member.name]

        tar.extract(member, path=path)


def rename_files_in_directory(directory):
    ann_path = "screen foto/dataset 2024-04-21 14_33_36/ann"
    img_path = "screen foto/dataset 2024-04-21 14_33_36/img"
    if os.path.exists(ann_path):
        print(f"Папка 'ann' найдена: {ann_path}")
        for count, filename in enumerate(os.listdir(ann_path), start=1):
            ext = os.path.splitext(filename)[1]
            new_name = f"{count}{ext}"
            os.rename(os.path.join(ann_path, filename), os.path.join(ann_path, new_name))
    else:
        print(f"Папка {ann_path} не найдена.")

    if os.path.exists(img_path):
        print(f"Папка 'img' найдена: {img_path}")
        for count, filename in enumerate(os.listdir(img_path), start=1):
            ext = os.path.splitext(filename)[1]
            new_name = f"{count}{ext}"
            os.rename(os.path.join(img_path, filename), os.path.join(img_path, new_name))
    else:
        print(f"Папка {img_path} не найдена.")


tar_path = "296658_US screen foto.tar"
extract_path = "screen foto"

os.makedirs(extract_path, exist_ok=True)

with tarfile.open(tar_path, "r") as tar:
    safe_extract(tar, path=extract_path)

rename_files_in_directory(extract_path)

print(f"Файлы переименованы в директории: {extract_path}/dataset/ann и {extract_path}/dataset/img")
