import os
import os.path
import os.path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data
import torch.utils.data
from sklearn.manifold import TSNE


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
    plt.close()


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
    plt.close()


def plot_confusion_matrix(cm, class_names, save_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediction')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    os.makedirs(os.path.join(save_dir, "output"), exist_ok=True)
    save_dir = os.path.join(save_dir, "output", "confusion_matrix.png")

    plt.savefig(save_dir, dpi=300)
    plt.close()


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
    plt.close()


def test_generator(img_list, save_folder):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    if save_folder is not None:
        # Lưu animation thành file GIF
        ani.save(f"{save_folder}/animation_generator.gif", writer="pillow", fps=1)
    # display(HTML(ani.to_jshtml()))
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, optimal_idx, optimal_threshold, save_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black',
                label='Optimal threshold = %0.2f' % optimal_threshold)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    # plt.grid(alpha=0.3)

    os.makedirs(os.path.join(save_dir, "output"), exist_ok=True)
    save_dir = os.path.join(save_dir, "output", "ROC_AUC.png")
    plt.show()
    plt.savefig(save_dir, dpi=300)
    plt.close()


def calculator_metric(cm):
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return far, frr, acc