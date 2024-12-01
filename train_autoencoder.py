import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from dataset import RGBDGraspDataset
from model import DepthAutoencoder
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
model = DepthAutoencoder().to(device)

lr = 1e-3
num_epochs = 10
batch_size = 64

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for _, depth_maps, _ in tqdm(train_loader, desc="Training"):
        inputs = depth_maps.to(device)
        
        optimizer.zero_grad()
        _, reconstructed = model(inputs)
        loss = criterion(reconstructed, inputs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Training Loss: {running_loss / len(train_loader):.4f}")


torch.save(model.state_dict(), "autoencoder.pt")
