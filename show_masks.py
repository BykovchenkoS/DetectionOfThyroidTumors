import json
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def decode_base64_to_image(base64_data):
    try:
        img_data = base64.b64decode(base64_data)
        img = Image.open(BytesIO(img_data))
        return np.array(img)
    except Exception as e:
        print(f"Ошибка при декодировании Base64: {e}")
        return None


# Загрузка JSON-файла
json_file = 'screen foto/dataset 2024-04-21 14_33_36/clear_ann/4.json'
with open(json_file, 'r') as f:
    json_data = json.load(f)

# Загрузить основное изображение
image = cv2.imread('screen foto/dataset 2024-04-21 14_33_36/img/4.jpg')

# Обрабатываем все объекты с масками
for obj in json_data['objects']:
    if 'bitmap' in obj:  # Если объект содержит маску
        mask_base64 = obj['bitmap']['data']
        position = obj['bitmap']['origin']

        # Декодируем маску из Base64
        mask = decode_base64_to_image(mask_base64)

        if mask is not None:
            # Получаем размеры маски
            mask_height, mask_width = mask.shape[:2]

            # Координаты для наложения маски на изображение
            y, x = position

            # Проверяем, что маска помещается в пределах изображения
            if y + mask_height <= image.shape[0] and x + mask_width <= image.shape[1]:
                # Наложение маски на изображение
                image[y:y + mask_height, x:x + mask_width] = mask
            else:
                print(f"Маска для объекта {obj['classTitle']} не помещается на изображение.")

# Отобразить результат
cv2.imshow("Image with Masks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
