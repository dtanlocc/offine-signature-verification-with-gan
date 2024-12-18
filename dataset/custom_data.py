import random
import os
import pandas as pd
from sklearn.model_selection import KFold

def create_pairs(signature_genuine: list, signature_forged: list):
    """
    Tạo cặp chữ ký (genuine, genuine) và (genuine, forged)
    """
    all_genuine_pairs = [(signature_genuine[i], signature_genuine[j])
                         for i in range(len(signature_genuine))
                         for j in range(1 + i, len(signature_genuine))]

    all_forged_pairs = [(genuine, forged) for genuine in signature_genuine for forged in signature_forged]

    return all_genuine_pairs, all_forged_pairs


def split_into_folds(genuine_pairs, forged_pairs, n_splits=5):
    """
    Chia các cặp chữ ký (genuine và forged) thành n_fold ngẫu nhiên và đảm bảo cân bằng nhãn
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_data = {i: {'genuine': [], 'forged': []} for i in range(n_splits)}  # Dictionary để lưu các fold với nhãn riêng biệt

    # Chia cặp genuine vào các fold
    for fold_idx, (_, test_idx) in enumerate(kf.split(genuine_pairs)):
        fold_data[fold_idx]['genuine'].extend([genuine_pairs[idx] for idx in test_idx])

    # Chia cặp forged vào các fold
    for fold_idx, (_, test_idx) in enumerate(kf.split(forged_pairs)):
        fold_data[fold_idx]['forged'].extend([forged_pairs[idx] for idx in test_idx])

    # Cân bằng số lượng genuine và forged trong mỗi fold
    for fold_idx in fold_data:
        num_genuine = len(fold_data[fold_idx]['genuine'])
        num_forged = len(fold_data[fold_idx]['forged'])

        if num_genuine != num_forged:
            min_count = min(num_genuine, num_forged)
            fold_data[fold_idx]['genuine'] = fold_data[fold_idx]['genuine'][:min_count]
            fold_data[fold_idx]['forged'] = fold_data[fold_idx]['forged'][:min_count]

    return fold_data


# Khởi tạo DataFrame
paths = ['../data/CEDAR/CEDAR/', '../data/BHSig260-Hindi/BHSig260-Hindi/', '../data/BHSig260-Bengali/BHSig260-Bengali']
outputs = ['../data/CEDAR/data.csv', '../data/BHSig260-Hindi/data.csv', '../data/BHSig260-Bengali/data.csv']

for path, output in zip(paths, outputs):
    signers = os.listdir(path)
    df = pd.DataFrame(columns=['user', 'image_1', 'image_2', 'label'])

    all_signers_folds = {i: {'genuine': [], 'forged': []} for i in range(5)}  # Lưu các fold cho tất cả signer

    # List to track all pairs from all folds for duplicate checking
    all_pairs = []

    for signer in signers:
        signatures_genuine = []
        signatures_forged = []
        signer_path = os.path.join(path, signer)
        all_signature_path = os.listdir(signer_path)

        for signature_path in all_signature_path:
            if 'forger' in signature_path or 'F' in signature_path:
                signatures_forged.append(f"{signer}/{signature_path}")
            if 'ori' in signature_path or 'G' in signature_path:
                signatures_genuine.append(f"{signer}/{signature_path}")

        # Tạo các cặp chữ ký genuine và forged cho signer này
        genuine_pairs, forged_pairs = create_pairs(signatures_genuine, signatures_forged)

        # Chia các cặp thành 5 fold ngẫu nhiên và đảm bảo cân bằng nhãn
        folds = split_into_folds(genuine_pairs, forged_pairs, n_splits=5)

        # Gộp các fold của signer này vào fold chung
        for fold_idx in range(5):
            all_signers_folds[fold_idx]['genuine'].extend(folds[fold_idx]['genuine'])
            all_signers_folds[fold_idx]['forged'].extend(folds[fold_idx]['forged'])

            # Thêm các cặp vào danh sách tổng
            all_pairs.extend(folds[fold_idx]['genuine'])
            all_pairs.extend(folds[fold_idx]['forged'])

    # Kiểm tra trùng lặp các cặp giữa các fold
    duplicate_pairs = set()
    seen_pairs = set()
    for pair in all_pairs:
        if pair in seen_pairs:
            duplicate_pairs.add(pair)
        else:
            seen_pairs.add(pair)

    if duplicate_pairs:
        print(f"Found duplicate pairs across folds: {duplicate_pairs}")
    else:
        print("No duplicate pairs found across folds.")

    # Kiểm tra tổng số cặp trước và sau khi chia fold
    total_genuine = sum(len(all_signers_folds[fold_idx]['genuine']) for fold_idx in range(5))
    total_forged = sum(len(all_signers_folds[fold_idx]['forged']) for fold_idx in range(5))
    print(f"Total genuine pairs: {total_genuine}")
    print(f"Total forged pairs: {total_forged}")
    print(f"Total pairs: {total_genuine + total_forged}")

    # Sau khi chia fold cho tất cả các signer, tạo dataframe từ các fold
    for fold_idx in range(5):
        fold_data_genuine = all_signers_folds[fold_idx]['genuine']
        fold_data_forged = all_signers_folds[fold_idx]['forged']
    # Cặp ảnh thật thì label là 1, ảnh giả thì label là 0
        rows = []
        for pair in fold_data_genuine:
            user = pair[0].split('/')[0]
            image_1 = pair[0]
            image_2 = pair[1]
            label = 1
            rows.append({'user': user, 'image_1': image_1, 'image_2': image_2, 'label': label})

        for pair in fold_data_forged:
            user = pair[0].split('/')[0]
            image_1 = pair[0]
            image_2 = pair[1]
            label = 0
            rows.append({'user': user, 'image_1': image_1, 'image_2': image_2, 'label': label})

        fold_df = pd.DataFrame(rows)
        print(f'_fold_{fold_idx + 1}')
        print(len(fold_df))
        print(f"Label: 0 có {fold_df['label'].value_counts().get(0, 0)}")
        print(f"Label: 1 có {fold_df['label'].value_counts().get(1, 0)}")
        print(len(fold_df['user'].unique()))
        fold_df.to_csv(f"{output.replace('.csv', f'_fold_{fold_idx + 1}.csv')}", index=False)

    print(f"Data for {path} has been split into 5 folds and saved.")
