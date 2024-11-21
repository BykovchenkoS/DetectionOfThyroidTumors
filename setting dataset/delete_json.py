import os
import json
import shutil

source_folder = "screen foto/dataset 2024-04-21 14_33_36/ann"
destination_folder = "screen foto/dataset 2024-04-21 14_33_36/clear_ann"

os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(source_folder, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if data.get('objects'):
                    shutil.copy(file_path, destination_folder)
                    print(f'Копируем файл: {filename}')
                else:
                    print(f'Пропускаем файл: {filename} (пустое поле "objects")')
            except json.JSONDecodeError:
                print(f'Ошибка чтения JSON в файле: {filename}')
