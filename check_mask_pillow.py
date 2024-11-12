import matplotlib.pyplot as plt
import cv2

image_path = 'screen foto/dataset 2024-04-21 14_33_36/img/4.jpg'
image = cv2.imread(image_path)

mask_path = 'screen foto/dataset 2024-04-21 14_33_36/masks/4_mask.png'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.axis('off')

masked_image = cv2.bitwise_and(image, image, mask=mask)
masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
plt.subplot(1, 3, 3)
plt.imshow(masked_image_rgb)
plt.title("Masked Image")
plt.axis('off')

plt.show()