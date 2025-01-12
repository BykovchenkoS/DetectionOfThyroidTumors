import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CocoDetection
from torch.optim import Adam

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = CocoDetection(
    root='path_to_train_images',  # Путь к изображениям для тренировки
    annFile='path_to_train_annotations.json',  # Путь к аннотациям для тренировки
    transform=transform
)

val_dataset = CocoDetection(
    root='path_to_val_images',  # Путь к изображениям для валидации
    annFile='path_to_val_annotations.json',  # Путь к аннотациям для валидации
    transform=transform
)


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.train()

optimizer = Adam(model.parameters(), lr=0.001)


def train(model, train_loader, val_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            train_loss += losses.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                loss_dict = model(images, targets)
                val_losses = sum(loss for loss in loss_dict.values())
                val_loss += val_losses.item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")


train(model, train_loader, val_loader, optimizer)
