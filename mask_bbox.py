from PIL import Image
import numpy as np

mask = Image.open('screen foto/dataset 2024-04-21 14_33_36/masks/2_Carotis_1556365529.png').convert('1')  # '1' для бинарного изображения
mask_array = np.array(mask)

origin_x = 12
origin_y = 145

object_pixels = np.where(mask_array == 1)

# Нахождение минимальных и максимальных координат с учетом origin
xmin = np.min(object_pixels[1]) + origin_x
xmax = np.max(object_pixels[1]) + origin_x
ymin = np.min(object_pixels[0]) + origin_y
ymax = np.max(object_pixels[0]) + origin_y

bbox = (xmin, ymin, xmax, ymax)
print("Bounding Box:", bbox)

width = xmax - xmin
height = ymax - ymin
coco_bbox = [xmin, ymin, width, height]
print("COCO Bounding Box:", coco_bbox)
