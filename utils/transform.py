import random
import torch
import numpy as np


class Resize(object):
    """
    Resize the image to a given size.
    :param output_size: The target size tuple (256, 256).
    """
    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, sample):
        return self._resize_image(sample)

    def _resize_image(self, img):
        # Resize image
        w, h = img.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = img.resize((new_w, new_h))  # Resize using PIL
        return img


class RandomCrop(object):
    """
    Randomly crop the image to the given size.
    :param output_size: Desired output size for the crop (tuple).
    """
    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, sample):
        return self._random_crop(sample)

    def _random_crop(self, img):
        w, h = img.size
        new_w, new_h = self.output_size

        if w == new_w and h == new_h:
            return img  # No cropping needed if size is already matched

        # Randomly determine the crop box
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img_cropped = img.crop((left, top, left + new_w, top + new_h))

        return img_cropped


class StandardizeImage(object):
    """Standardize image using predefined mean and std values for grayscale images."""

    def __init__(self):
        self.mean = np.array([0.5])  # Predefined mean for grayscale image
        self.std = np.array([0.5])  # Predefined std for grayscale image

    def __call__(self, img):
        """
        Chuẩn hóa ảnh sử dụng mean và std cố định.

        :param img: Input image (torch.Tensor).
                    Dự kiến đầu vào là ảnh grayscale có dạng (C, H, W) sau khi đã chuyển sang tensor.
        :return: Ảnh chuẩn hóa có cùng dạng như đầu vào.
        """
        # Nếu ảnh có số chiều là (C, H, W) thì thực hiện chuẩn hóa theo từng kênh
        mean = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)  # (C, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)  # (C, 1, 1)

        # Tránh chia cho 0
        std[std == 0] = 1

        # Chuẩn hóa ảnh: (img - mean) / std
        standardized_img = (img - mean) / std

        return standardized_img