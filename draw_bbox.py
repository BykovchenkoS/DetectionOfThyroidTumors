from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

image_path = "dataset_node/img/6.jpg"

img = Image.open(image_path)

bbox = [420, 267, 108, 108]

x, y, width, height = bbox
draw = ImageDraw.Draw(img)
draw.rectangle([x, y, x + width, y + height], outline="red", width=3)

plt.imshow(img)
plt.show()
