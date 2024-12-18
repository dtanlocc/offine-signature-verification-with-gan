import torch
import torch.nn.functional as F


def triplet_loss(pos_dist, neg_dist, margin=1.0):
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def contrastive_loss(y_pred, y_true, margin=1.0):
    # Tính bình phương của y_pred (khoảng cách Euclidean)
    square_pred = torch.pow(y_pred, 2)

    # Tính phần margin (nếu y_pred lớn hơn margin, thì mất mát sẽ là (y_pred - margin)^2)
    margin_square = torch.pow(torch.clamp(margin - y_pred, min=0), 2)

    # Hàm loss cho các cặp ảnh giống nhau (y_true=1) và khác nhau (y_true=0)
    loss = torch.mean(y_true * square_pred + (1 - y_true) * margin_square)

    return loss
