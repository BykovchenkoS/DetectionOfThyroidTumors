from PIL import Image
import numpy as np

mask = Image.open('../dataset_coco_neuro_2/masks/100_Node_1554990432.png').convert('1')
mask_array = np.array(mask)

origin_x = 12
origin_y = 13

object_pixels = np.where(mask_array == 1)

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
