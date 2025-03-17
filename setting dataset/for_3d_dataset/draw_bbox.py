from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

image_path = "dataset_coco_neuro_3/img/121.jpg"
mask_path = "dataset_coco_neuro_3/masks/121.png"

img = Image.open(image_path)
mask = Image.open(mask_path)

bbox = [205.95744680851064, 295.93617021276594, 273.36170212765956, 116.08510638297872]

mask_array = np.array(mask)
mask_rgba = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)

mask_rgba[..., 0] = 255
mask_rgba[..., 3] = np.where(mask_array > 0, 128, 0)

colored_mask = Image.fromarray(mask_rgba, mode="RGBA")

img_with_mask = img.copy()
img_with_mask.paste(colored_mask, (0, 0), mask=colored_mask)

x, y, width, height = bbox
draw = ImageDraw.Draw(img_with_mask)
draw.rectangle([x, y, x + width, y + height], outline="red", width=3)

plt.imshow(img_with_mask)
plt.axis("off")
plt.show()
