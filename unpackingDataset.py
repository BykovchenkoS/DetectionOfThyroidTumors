import tarfile
import os

def safe_extract(tar, path="."):
    for member in tar.getmembers():
        # меняем двоеточия в именах файлов на подчеркивания
        member.name = member.name.replace(":", "_")
        tar.extract(member, path=path)

tar_path = "296658_US screen foto.tar"
extract_path = "screen foto"

os.makedirs(extract_path, exist_ok=True)

with tarfile.open(tar_path, "r") as tar:
    safe_extract(tar, path=extract_path)

print(f"Проект распакован в директорию: {extract_path}")