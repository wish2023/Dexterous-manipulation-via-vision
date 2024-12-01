import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from dataset import RGBDGraspDataset
from model import ResNet18RGBD
from utils import get_device


datasets = [
    {
        'image_path': '../data/DeepGrasping_JustImages',
        'depth_path': '../data/DEPTH_DeepGrasping_JustImages',
        'anno_path': '../data/DeepGrasping_Anno',
        'image_subdirs': [f'{i:02}' for i in range(1, 11)],
    },
    {
        'image_path': '../data/Imagenet',
        'depth_path': '../data/DEPTH_Imagenet',
        'anno_path': '../data/Anno_ImageNet.json',
    },
    {
        'image_path': '../data/HandCam',
        'depth_path': '../data/DEPTH_HandCam',
        'anno_path': '../data/Anno_HandCam4.json',
    }
]

train_dataset = RGBDGraspDataset(datasets)
device = get_device()
model = ResNet18RGBD().to(device)
for name, param in model.named_parameters():
        if "fc" in name or 'model.conv1.weight' == name:
              param.requires_grad = True
        else:
              param.requires_grad = False



lr = 1e-3
num_epochs = 10
batch_size = 8

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=lr
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for rgb_maps, depth_maps, labels in tqdm(train_loader, desc="Training"):
        rgb_maps, depth_maps = rgb_maps.to(device), depth_maps.to(device)
        labels = labels.to(device)
        inputs = torch.cat((rgb_maps, depth_maps), dim=1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        model.model.conv1.weight.grad[:, :3, :, :] = 0
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f"Training Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

