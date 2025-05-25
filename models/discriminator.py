import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, output_dim=128):
        super(Discriminator, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.Dropout(0.3),

            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.Dropout(0.3),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, output_dim) # Kích thước đầu ra là output_dim
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        euclidean_distance = F.pairwise_distance(out1, out2)
        return euclidean_distance
