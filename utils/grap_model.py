from torch.utils.tensorboard import SummaryWriter
from models.generator import Generator
from models.discriminator import Discriminator
import torch

# Khởi tạo SummaryWriter
writer = SummaryWriter('D:/LVTN/images/discriminator')
model_g = Generator(z_dim=64, img_height=155, img_width=220, channels=1)
model_d = Discriminator()
# Tạo dummy input
z = torch.randn(1, 64)
img_1 = torch.randn(1, 1, 155, 220)
img_2 = torch.randn(1, 1, 155, 220)

# Ghi model vào TensorBoard
# writer.add_graph(model_g, (z, img_1))
writer.add_graph(model_d, (img_1, img_2))

# Đóng writer
writer.close()