import pandas as pd
from torch.utils.data import DataLoader
from dataset.sign_dataset import SignatureDataset
from typing import List
import pandas as pd


def create_train_test_loaders(csv_files: List, path_root: str, transform=None, batch_size=32, idx_fold_test=0):
    train_files = [csv_files[i] for i in range(len(csv_files)) if i != idx_fold_test]

    # Kết hợp các tệp train lại với nhau
    train_data = pd.concat([pd.read_csv(file) for file in train_files], ignore_index=True)

    # Đọc tệp CSV cho test data
    test_data = pd.read_csv(csv_files[idx_fold_test])

    # Tạo dataset cho train và test
    train_dataset = SignatureDataset(path_root=path_root, data_frame=train_data, transform=transform)
    print(f"Train dataset len: {len(train_dataset)}")
    test_dataset = SignatureDataset(csv_files[idx_fold_test], path_root, transform=transform)
    print(f"Test dataset len: {len(test_dataset)}")

    # Tạo DataLoader cho train và test
    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(test_data['user'].unique())
