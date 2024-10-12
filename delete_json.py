import os
import json
import shutil

source_folder = "screen foto/dataset 2024-04-21 14_33_36/ann"
# Путь к новой папке для сохранения только верных JSON файлов
destination_folder = "screen foto/dataset 2024-04-21 14_33_36/clear_ann"


# Создаем папку для валидных файлов, если она еще не существует
os.makedirs(destination_folder, exist_ok=True)

# Проходим по всем файлам в исходной папке
for filename in os.listdir(source_folder):
    # Проверяем, что файл имеет расширение .json
    if filename.endswith('.json'):
        file_path = os.path.join(source_folder, filename)

        # Читаем JSON файл
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # Проверяем, что в "objects" есть данные
                if data.get('objects'):
                    # Копируем файл в новую папку
                    shutil.copy(file_path, destination_folder)
                    print(f'Копируем файл: {filename}')
                else:
                    print(f'Пропускаем файл: {filename} (пустое поле "objects")')
            except json.JSONDecodeError:
                print(f'Ошибка чтения JSON в файле: {filename}')
