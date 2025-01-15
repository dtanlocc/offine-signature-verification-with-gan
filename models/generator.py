import torch
import torch.nn as nn
import torch.nn.functional as F


# --- U-Net Generator --
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2)

        self.upsample_layer = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, skip=None):
        if self.upsample:
            x = self.upsample_layer(x)
            if skip is not None:
                diffY = skip.size()[2] - x.size()[2]
                diffX = skip.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
                x = torch.cat([x, skip], dim=1)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class Generator(nn.Module):
    def __init__(self, z_dim=64, img_height=155, img_width=220, channels=1):
        super().__init__()

        # Encoder
        self.enc_conv1 = UNetBlock(channels, 16)
        self.enc_pool1 = nn.MaxPool2d(kernel_size=2)  # 77 x 110
        self.enc_conv2 = UNetBlock(16, 32)
        self.enc_pool2 = nn.MaxPool2d(kernel_size=2)  # 38 x 55
        self.enc_conv3 = UNetBlock(32, 64)
        self.enc_pool3 = nn.MaxPool2d(kernel_size=2)  # 19 x 27
        self.enc_conv4 = UNetBlock(64, 128)
        self.enc_pool4 = nn.MaxPool2d(kernel_size=2)  # 9 x 13

        # Bottleneck
        self.bottleneck = UNetBlock(128, 256)

        # Decoder
        self.dec_upsample1 = UNetBlock(384, 128, upsample=True)
        self.dec_upsample2 = UNetBlock(192, 64, upsample=True)
        self.dec_upsample3 = UNetBlock(96, 32, upsample=True)
        self.dec_upsample4 = UNetBlock(48, 16, upsample=True)

        # Output layer
        self.dec_conv = nn.Conv2d(16, channels, kernel_size=3, padding=1)
        self.dec_tanh = nn.Tanh()

        # Noise
        self.noise_fc = nn.Linear(z_dim, 256)
        self.noise_relu = nn.LeakyReLU(0.2)

    def forward(self, z, img):
        # Encoder
        enc1 = self.enc_conv1(img)
        enc1_pool = self.enc_pool1(enc1)
        enc2 = self.enc_conv2(enc1_pool)
        enc2_pool = self.enc_pool2(enc2)
        enc3 = self.enc_conv3(enc2_pool)
        enc3_pool = self.enc_pool3(enc3)
        enc4 = self.enc_conv4(enc3_pool)
        enc4_pool = self.enc_pool4(enc4)

        # Bottleneck
        b = self.bottleneck(enc4_pool)

        # Add noise to the bottleneck
        noise = self.noise_relu(self.noise_fc(z))
        noise = noise.view(z.size(0), 256, 1, 1)  # Reshape noise to match bottleneck spatial dimensions
        b = b + noise

        # Decoder
        dec4 = self.dec_upsample1(b, enc4)
        dec3 = self.dec_upsample2(dec4, enc3)
        dec2 = self.dec_upsample3(dec3, enc2)
        dec1 = self.dec_upsample4(dec2, enc1)

        # Output layer
        out = self.dec_tanh(self.dec_conv(dec1))
        return out
