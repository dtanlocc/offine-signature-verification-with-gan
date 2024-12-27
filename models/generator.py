import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, nz, ngf=64):  # ngf: số lượng feature maps của Generator
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(nz, ngf * 8 * 5 * 7)  # Đầu vào là vector nhiễu

        self.main = nn.Sequential(
            # Khối 1: 5x7 -> 10x14
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Khối 2: 10x14 -> 20x28
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Khối 3: 20x28 -> 40x56
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Khối 4: 40x56 -> 80x112
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Khối 5: 80x112 -> 160x224
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Khối 6: Điều chỉnh kích thước về 155x220
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False), # Kích thước không đổi, dùng Conv2d thay vì ConvTranspose2d
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 1, (6, 7), 1, (2, 3), bias=False), # Kích thước từ 160x224 -> 155x220
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc1(z.view(z.size(0), -1))
        x = x.view(x.size(0), -1, 5, 7)  # Reshape phù hợp với đầu vào ConvTranspose2d
        return self.main(x)
