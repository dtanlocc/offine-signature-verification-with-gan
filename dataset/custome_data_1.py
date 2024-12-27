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
                         for j in range(i + 1, len(signature_genuine))]

    all_forged_pairs = [(genuine, forged) for genuine in signature_genuine for forged in signature_forged]

    return all_genuine_pairs, all_forged_pairs

def split_users_into_folds(users, n_splits=5):
    """
    Chia user thành các fold ngẫu nhiên.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    user_folds = {i: [] for i in range(n_splits)}
    for fold_idx, (_, test_idx) in enumerate(kf.split(users)):
        user_folds[fold_idx] = [users[idx] for idx in test_idx]
    return user_folds

# Khởi tạo DataFrame
paths = ['../data/CEDAR/CEDAR/', '../data/BHSig260-Hindi/BHSig260-Hindi/', '../data/BHSig260-Bengali/BHSig260-Bengali']
outputs = ['../data/CEDAR/fold_user/data.csv', '../data/BHSig260-Hindi/fold_user/data.csv', '../data/BHSig260-Bengali/fold_user/data.csv']

for path, output in zip(paths, outputs):
    signers = os.listdir(path)
    user_folds = split_users_into_folds(signers, n_splits=5)

    all_fold_data = {i: [] for i in range(5)}  # Lưu dữ liệu cho từng fold

    for fold_idx, users_in_fold in user_folds.items():
        fold_genuine_pairs = []
        fold_forged_pairs = []

        for signer in users_in_fold:
            signatures_genuine = []
            signatures_forged = []
            signer_path = os.path.join(path, signer)
            all_signature_path = os.listdir(signer_path)

            for signature_path in all_signature_path:
                if 'forger' in signature_path or 'F' in signature_path:
                    signatures_forged.append(f"{signer}/{signature_path}")
                if 'ori' in signature_path or 'G' in signature_path:
                    signatures_genuine.append(f"{signer}/{signature_path}")

            # Tạo các cặp chữ ký genuine và forged cho user này
            genuine_pairs, forged_pairs = create_pairs(signatures_genuine, signatures_forged)

            fold_genuine_pairs.extend(genuine_pairs)
            fold_forged_pairs.extend(forged_pairs)

        # Đảm bảo số lượng cặp genuine và forged bằng nhau
        min_pairs = min(len(fold_genuine_pairs), len(fold_forged_pairs))
        fold_genuine_pairs = random.sample(fold_genuine_pairs, min_pairs)
        fold_forged_pairs = random.sample(fold_forged_pairs, min_pairs)

        # Gộp các cặp vào fold data
        all_fold_data[fold_idx].extend([(pair[0], pair[1], 1) for pair in fold_genuine_pairs])  # Label 1 cho genuine
        all_fold_data[fold_idx].extend([(pair[0], pair[1], 0) for pair in fold_forged_pairs])  # Label 0 cho forged

    # Lưu dữ liệu từng fold vào file CSV
    for fold_idx in range(5):
        fold_data = all_fold_data[fold_idx]
        df = pd.DataFrame(fold_data, columns=['image_1', 'image_2', 'label'])
        print(f"Fold {fold_idx + 1}: {len(df)} samples")
        print(f"Label 0: {len(df[df['label'] == 0])}, Label 1: {len(df[df['label'] == 1])}")
        df.to_csv(f"{output.replace('.csv', f'_fold_{fold_idx + 1}.csv')}", index=False)

    print(f"Data for {path} has been split into 5 folds and saved.")