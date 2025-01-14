import torch
import matplotlib.pyplot as plt

mask_file = 'dataset_coco_neuro_1/train/masks_test/100_mask_2.pt'

mask = torch.load(mask_file)

plt.imshow(mask.numpy(), cmap='gray')
plt.title('Маска объекта')
plt.axis('off')
plt.show()