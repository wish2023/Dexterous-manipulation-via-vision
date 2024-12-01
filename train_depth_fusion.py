import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from dataset import RGBDGraspDataset
from model import ResNetDepthFusion
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

train_dataset = RGBDGraspDataset(datasets[:2])
test_dataset = RGBDGraspDataset([datasets[2]])
device = get_device()



model = ResNetDepthFusion().to(device)
for name, param in model.named_parameters():
        if "fc" in name:
                param.requires_grad = True
        else:
                param.requires_grad = False

# Hyperparameters
lr = 1e-3
num_epochs = 100

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=lr
)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

highest_accuracy = 0
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for rgb_maps, depth_maps, labels in tqdm(train_loader, desc="Training"):
        rgb_maps, depth_maps, labels = rgb_maps.to(device), depth_maps.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb_maps, depth_maps)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f"Training Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for rgb_maps, depth_maps, labels in tqdm(test_loader, desc="Validation"):
            rgb_maps, depth_maps, labels = rgb_maps.to(device), depth_maps.to(device), labels.to(device)
        
            outputs = model(rgb_maps, depth_maps)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        if val_accuracy > highest_accuracy:
            highest_accuracy = val_accuracy
            torch.save(model.state_dict(), "depth_fusion.pt")

        print(f"Validation accuracy: {val_accuracy:.2f}%")

