import torch

import torch.nn as nn
import torchvision.models as models

class ResNet50RGB(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet50RGB, self).__init__()
        
        self.model = models.resnet50(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VGG16RGB(nn.Module):
    def __init__(self, num_classes=5):
        super(VGG16RGB, self).__init__()
        
        self.model = models.vgg16(weights='DEFAULT')
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet50RGBD(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet50RGBD, self).__init__()
        
        self.model = models.resnet50(weights='DEFAULT')
        
        # Modify the first convolutional layer
        rgb_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=4,  # RGB + Depth
            out_channels=rgb_conv1.out_channels,
            kernel_size=rgb_conv1.kernel_size,
            stride=rgb_conv1.stride,
            padding=rgb_conv1.padding,
            bias=rgb_conv1.bias
        )
        
        # Initialize new weights for the depth channel
        with torch.no_grad():
            self.model.conv1.weight[:, :3, :, :] = rgb_conv1.weight  # Copy RGB weights

        # Modify the final fully connected layer for the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    


class DepthAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(DepthAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 1x224x224 -> 32x112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32x112x112 -> 64x56x56
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1), # 64x56x56 -> latent_dimx28x28
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(latent_dim * 28 * 28, latent_dim)  # Latent representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 28 * 28),
            nn.Unflatten(1, (latent_dim, 28, 28)),
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # To scale output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded




class ResNetDepthFusion(nn.Module):
    def __init__(self, autoencoder_weights='autoencoder.pt', resnet_weights='DEFAULT', latent_dim=512, num_classes=5):
        super(ResNetDepthFusion, self).__init__()

        self.resnet = models.resnet50(weights=resnet_weights)
        self.fc = nn.Linear(self.resnet.fc.in_features + latent_dim, num_classes)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.depth_autoencoder = DepthAutoencoder(latent_dim=latent_dim)
        self.depth_autoencoder.load_state_dict(torch.load(autoencoder_weights, map_location=torch.device('cpu')))

    def forward(self, rgb, depth):
        output1 = self.resnet(rgb).squeeze(-1).squeeze(-1)
        output2 = self.depth_autoencoder(depth)[0]

        output = torch.cat((output1, output2), 1)
        output = self.fc(output)

        return output