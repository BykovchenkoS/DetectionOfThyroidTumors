import cv2
import numpy as np

# Загрузка изображения
image_path = 'screen foto/dataset 2024-04-21 14_33_36/img/121.jpg'
image = cv2.imread(image_path)

# Загрузка маски (например, бинарная маска)
mask_path = 'screen foto/dataset 2024-04-21 14_33_36/masks/121_mask.png'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

masked_image = cv2.bitwise_and(image, image, mask=binary_mask)

cv2.imshow('Original Image', image)
cv2.imshow('Mask', binary_mask)
cv2.imshow('Masked Image', masked_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
