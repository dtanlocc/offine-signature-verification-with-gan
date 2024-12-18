import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, num_users, user_embedding_dim=128):
        super(Discriminator, self).__init__()

        # User Embedding layer
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)

        # ResNet50
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final fully connected layer

        # Fully connected layers to process the combined features
        self.fc = nn.Sequential(
            nn.Linear(2048 + user_embedding_dim, 1024),  # 2048 from ResNet + user_embedding_dim
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.up = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),  # Convert 1 channel to 3 channels
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),  # Resize image to 224x224
        )

    def forward_once(self, image, user_id):
        image = self.up(image)
        # Process image through ResNet50
        image_features = self.resnet(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten to (batch_size, 2048)

        # Get user embedding
        user_embedding = self.user_embedding(user_id)

        # Concatenate image features with user embedding
        combined_features = torch.cat((image_features, user_embedding), dim=1)

        return combined_features

    def forward(self, image_1, image_2, user_id):
        feature_image_1 = self.forward_once(image_1, user_id)
        feature_image_2 = self.forward_once(image_2, user_id)

        distance = F.pairwise_distance(feature_image_1, feature_image_2, p=2)
        return distance

