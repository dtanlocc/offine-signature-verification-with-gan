import json
import os

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils.meters import CatMeter


class BaseTrainer:
    def __init__(self, opt, fold, train_dataloader, test_dataloader):
        torch.manual_seed(opt.SEED)
        self.opt = opt
        if opt.USE_GPU:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.batch_size = opt.BATCH_SIZE
        self.epochs = opt.EPOCHS
        self.fold = fold
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.save_dir = None
        self.model_dir = None

    def _init_model(self):
        self.model = None

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def _init_optimizer(self):
        pass

    def _init_criterion(self):
        pass

    def _init_scheduler(self):
        pass

    def _create_folder(self, ):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def save_results(self, results):
        save_path = os.path.join(self.save_dir, f'results_fold{self.fold}.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {save_path}")

    def train(self):
        pass

    def _train_one_epoch(self, epoch):
        pass

    def _update_values(self, epoch):
        pass

    def test(self, epoch, tsne, flag, roc):
        self.load_model(epoch)
        print("Testing ...")
        return self.evaluate(epoch, self.model, tsne, flag, roc)

    def accuracy(self, distances, labels, step=0.001):
        dmax = max(distances).item()
        dmin = min(distances).item()
        nsame = torch.sum(labels == 1)
        ndiff = torch.sum(labels == 0)

        max_acc = 0
        best_thresh = 0.5

        for d in torch.arange(dmin, dmax + step, step):
            idx1 = distances <= d
            idx2 = distances > d

            tpr = float(torch.sum(labels[idx1] == 1)) / nsame
            tnr = float(torch.sum(labels[idx2] == 0)) / ndiff
            acc = 0.5 * (tpr + tnr)

            if acc > max_acc:
                max_acc = acc
                best_thresh = d

        return max_acc, best_thresh

    def evaluate(self, epoch, model, tsne=False, flag=False, roc=True):
        model.eval()

        test_distances_meter, test_labels_meter = CatMeter(), CatMeter()
        print("Calculating...")

        with torch.no_grad():
            for i, sample in enumerate(self.test_dataloader):
                image_1 = sample['image_1'].to(self.device)
                image_2 = sample['image_2'].to(self.device)
                label = sample['label'].to(self.device).float()

                dist = model(image_1, image_2)
                test_distances_meter.update(dist)
                test_labels_meter.update(label)

                del image_1, image_2, label, dist
                torch.cuda.empty_cache()

        test_distances = test_distances_meter.get_val()
        test_labels = test_labels_meter.get_val()
        # print(test_labels.shape)
        # print(test_distances.shape)
        # print(test_distances)
        accuracy, thresold = self.accuracy(test_distances, test_labels)

        test_distances_cpu = test_distances_meter.get_val_numpy()
        test_labels_cpu = test_labels_meter.get_val_numpy()
        predict_y = (test_distances_cpu < thresold.numpy()).astype(int)
        # print(test_labels_cpu.shape)
        # print(test_distances_cpu.shape)
        # print(predict_y.shape)

        accuracy = accuracy_score(test_labels_cpu, predict_y)

        # Tính confusion matrix
        cm = confusion_matrix(test_labels_cpu, predict_y)

        class_report = classification_report(test_labels_cpu, predict_y, output_dict=True)
        if flag:
            print(f"Accuracy at epoch {epoch}: {accuracy:.4f}")
            print("Confusion Matrix:")
            print(cm)
            print("Classification Report:")
            print(class_report)

        if tsne:
            # plot_tsne(train_distances, train_labels, self.save_dir, epoch)
            pass
        del test_labels
        del test_distances
        torch.cuda.empty_cache()

        return accuracy, cm, class_report

    def save_model(self, epoch):
        pass

    def load_model(self, epoch):
        pass