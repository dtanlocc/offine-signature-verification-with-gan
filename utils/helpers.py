from models.discriminator import Discriminator
from models.generator import Generator
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import torch
import torch.nn as nn


def load_model(opt, device):
    print('Init model Discriminator')
    model_d = Discriminator(num_users=opt.NUM_USERS + 1, user_embedding_dim=opt.NUM_EMBEDDING).to(device)
    print('Init model Generator')
    model_g = Generator(nz=opt.NZ, num_users=opt.NUM_USERS + 1, user_embedding_dim=opt.NUM_EMBEDDING).to(device)
    model_g.apply(weights_init)
    return model_d, model_g


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_tsne(data, labels, save_dir, epoch):
    print("PLOTING TSNE ...")
    palette = sns.color_palette("bright", len(np.unique(labels)))
    tsne_output = TSNE(n_components=1).fit_transform(data)
    # print(n)
    plt.figure(figsize=(16, 10))

    sns.scatterplot(x=tsne_output[:, 0], y=labels, hue=labels, palette=palette, legend=False)

    os.makedirs(os.path.join(save_dir, "output"), exist_ok=True)
    save_dir = os.path.join(save_dir, "output", "tsne_" + str(epoch) + ".png")

    plt.savefig(save_dir, dpi=300)


def visualize_metrics_seaborn(classification_reports, save_dir):
    labels = [key for key in classification_reports.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]

    # Lấy dữ liệu cho recall, precision và f1-score
    recall = [classification_reports[label]['recall'] for label in labels]
    precision = [classification_reports[label]['precision'] for label in labels]
    f1_score = [classification_reports[label]['f1-score'] for label in labels]
    acc = classification_reports["accuracy"]

    # Tạo biểu đồ
    x = np.arange(len(labels))  # vị trí của các label
    width = 0.2  # chiều rộng của các cột

    fig, ax = plt.subplots(figsize=(10, 6))

    # Vẽ các cột cho mỗi loại chỉ số
    bars1 = ax.bar(x - width, precision, width, label='Precision')
    bars2 = ax.bar(x, recall, width, label='Recall')
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score')

    # Thêm các nhãn và tiêu đề
    ax.set_xlabel('Labels')
    ax.set_ylabel('Scores')
    ax.set_title(f'Acc: {acc}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Hiển thị giá trị trên đầu các cột
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    os.makedirs(os.path.join(save_dir, "output"), exist_ok=True)
    save_dir = os.path.join(save_dir, "output", "metric.png")

    plt.savefig(save_dir, dpi=300)


def plot_confusion_matrix(cm, class_names, save_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediction')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    os.makedirs(os.path.join(save_dir, "output"), exist_ok=True)
    save_dir = os.path.join(save_dir, "output", "confusion_matrix.png")

    plt.savefig(save_dir, dpi=300)


def plot_loss(train_losses, label, save_dir):
    plt.figure(figsize=(10, 5))
    # Chuyển tensor sang CPU và NumPy trước khi vẽ
    if isinstance(train_losses, torch.Tensor):
        train_losses = train_losses.cpu().numpy()

    plt.plot(train_losses, label=label, marker='o')
    plt.title('Loss during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs(os.path.join(save_dir, "output"), exist_ok=True)
    save_dir = os.path.join(save_dir, "output", "confusion_matrix.png")

    plt.savefig(save_dir, dpi=300)

def calculator_metric(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    far = fp / (fp + tn)  # FAR
    frr = fn / (fn + tp)  # FRR
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return far, frr, accuracy