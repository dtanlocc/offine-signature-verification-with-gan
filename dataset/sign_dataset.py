""""
    Tạo dataset từ file data.csv, phù hợp cho đầu vào model
    data.csv lưu path image


"""
import os.path
from PIL import Image
import pandas as pd
import torch.utils.data


class SignatureDataset(torch.utils.data.Dataset):
    """" Signature Dataset. """

    def __init__(self, csv_file: str = None, path_root: str = None, transform=None, data_frame=None):
        """
        :param csv_file: path to the CSV file with image paths and labels.
        :param path_root: Directory with all images.
        :param transform: Transform to be applied on a sample.
        """
        self.data_frame = data_frame
        self.path_root = path_root
        self.transform = transform
        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
        elif data_frame is not None:
            self.data = data_frame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data.loc[idx, 'user']
        image_1_path = os.path.join(self.path_root,
                                    self.data.loc[idx, 'image_1'])

        image_2_path = os.path.join(self.path_root,
                                    self.data.loc[idx, 'image_2'])

        label = self.data.loc[idx, 'label']

        image_1 = Image.open(image_1_path).convert('L')
        image_2 = Image.open(image_2_path).convert('L')

        sample = {'user': user, 'image_1': image_1, 'image_2': image_2, 'label': label}

        if self.transform:
            sample['image_1'] = self.transform(sample['image_1'])
            sample['image_2'] = self.transform(sample['image_2'])
            # sample = self.transform(sample)
        return sample
