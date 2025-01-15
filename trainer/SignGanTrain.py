import os
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm.notebook import tqdm

from models.discriminator import Discriminator
from models.generator import Generator
from trainer.BaseTrain import BaseTrainer
from utils.loss_function import TripletLoss, ContrastiveLoss
from utils.helpers import test_generator, calculator_metric, visualize_metrics_seaborn, plot_confusion_matrix
from utils.meters import CatMeter, AverageMeter


class SignGanTrainer(BaseTrainer):
    def __init__(self, opt, fold, train_dataloader, test_dataloader):
        super().__init__(opt, fold, train_dataloader, test_dataloader)
        self.img_list = []
        self.fixed_noise = torch.randn(opt.BATCH_SIZE, opt.NZ, device=self.device)
        self.save_dir = opt.SAVE_DIR_GAN
        self.model_dir = opt.MODEL_DIR_GAN
        self.accuracies = []
        self.train_g_losses = []
        self.train_d_losses = []
        self.results = {
            'GAN': {
                f'fold_{self.fold}': {
                    'frr': [],
                    'far': [],
                    'acc': [],
                    'loss_d': [],
                    'loss_g': []
                }
            }
        }

        self._init_model()
        self._init_criterion()
        self._init_optimizer()
        self._init_scheduler()
        self._create_folder()
        sample = next(iter(self.test_dataloader))
        self.fixed_image_1 = sample['image_1'].to(self.device)

        self.img_list.append(vutils.make_grid(self.fixed_image_1.cpu(), padding=2, normalize=True))

    def _init_model(self):
        self.model_g = Generator().to(self.device)
        self.model_d = Discriminator().to(self.device)
        self.model_d.apply(self.weights_init_normal)
        self.model_d.train()
        self.model_g.train()

    def _init_optimizer(self):
        self.optimizer_g = optim.Adam(self.model_g.parameters(), lr=5e-5, betas=(0.5, 0.9), weight_decay=1e-4)
        self.optimizer_d = optim.Adam(self.model_d.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)

    def _init_criterion(self):
        self.triplet_loss = TripletLoss(margin=1.0)
        self.contrastive_loss = ContrastiveLoss(margin=1.0)
        self.mse_loss = nn.MSELoss()

    def _init_scheduler(self):
        self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_g, mode='min', factor=0.5, patience=5)
        self.scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_d, mode='min', factor=0.5, patience=5)
        # pass

    def train(self):
        best_accuracy = 0.0
        best_epoch = 0

        print('START TRAINING.....')
        for epoch in range(self.epochs):
            self.model_d.train()
            self.model_g.train()
            train_dis_loss, train_gen_loss, accuracy, cm, classification_report_dict = self._train_one_epoch(epoch)
            self.train_g_losses.append(train_gen_loss)
            self.train_d_losses.append(train_dis_loss)
            self.accuracies.append(accuracy)

            self.results['GAN'][f'fold_{self.fold}']['loss_d'].append(train_dis_loss.item())
            self.results['GAN'][f'fold_{self.fold}']['loss_g'].append(train_gen_loss.item())
            test_generator(self.img_list, save_folder=None)
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                self.save_model(epoch)
        accuracy, cm, classification_report_dict = self.test(best_epoch, tsne=False, flag=True, roc=True)
        far, frr, acc = calculator_metric(cm)
        self.results['GAN'][f'fold_{self.fold}']['frr'].append(frr)
        self.results['GAN'][f'fold_{self.fold}']['far'].append(far)
        visualize_metrics_seaborn(classification_report_dict, self.save_dir + '/fold_' + str(self.fold))
        plot_confusion_matrix(cm, class_names=['0', '1'], save_dir=self.save_dir + '/fold_' + str(self.fold))
        print(f'Best Accuracy: {best_accuracy:.4f}, epoch: {best_epoch}')
        self.results['GAN'][f'fold_{self.fold}']['acc'].append(best_accuracy)
        self.save_results(self.results)

    def _train_one_epoch(self, epoch):
        self.d_loss_meter = AverageMeter()
        self.g_loss_meter = AverageMeter()

        progress_bar = tqdm(self.train_dataloader)
        for i, sample in enumerate(progress_bar):
            # Dữ liệu đầu vào
            self.image_1 = sample['image_1'].to(self.device)
            self.image_2 = sample['image_2'].to(self.device)
            self.label = sample['label'].to(self.device).float()

            z = torch.randn(self.image_1.size(0), self.opt.NZ, device=self.device)

            # Tạo ảnh giả
            self.fake_image_2 = self.model_g(z, self.image_1)

            # Train Discriminator
            d_loss = self.train_d()

            # Train Generator
            g_loss = self.train_g()

            self.d_loss_meter.update(d_loss)
            self.g_loss_meter.update(g_loss)

            progress_bar.set_description(
                f"EPOCH [{epoch}/{self.epochs}]: D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}"
            )

            del self.image_1, self.image_2, self.fake_image_2, z, self.label
            torch.cuda.empty_cache()

        with torch.no_grad():
            fake = self.model_g(self.fixed_noise, self.fixed_image_1).to(self.device).detach().cpu()
        self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # Ghi nhận loss mean
        d_loss_avg, _ = self.d_loss_meter.value()
        g_loss_avg, _ = self.g_loss_meter.value()

        accuracy, cm, classification_report_dict = self.evaluate(epoch, self.model_d, tsne=False, flag=False, roc=True)
        self.scheduler_d.step(d_loss_avg)
        self.scheduler_g.step(g_loss_avg)

        # Ghi nhận loss mean
        print(
            f"EPOCH {epoch}:\tTrain dis loss: {d_loss_avg:.4f}\tTrain gen loss: {g_loss_avg:.4f}\tAccuracy: {accuracy:.4f}")
        log_file = os.path.join(self.save_dir, f'train_log_fold{self.fold}.txt')
        with open(log_file, 'a') as log:
            log.write(
                f"Epoch {epoch}:\tTrain dis loss: {d_loss_avg:.4f}\tTrain gen loss: {g_loss_avg:.4f}\tAccuracy: {accuracy:.4f}\n")
        return d_loss_avg, g_loss_avg, accuracy, cm, classification_report_dict

    def train_d(self):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_d.zero_grad()
        output_real = self.model_d(self.image_1, self.image_2).view(-1)
        # label_real = torch.full((self.batch_size,), 0, dtype=torch.float, device=self.device) # Label 0 cho real
        errD_real = self.contrastive_loss(output_real, self.label)

        # Train with all-fake batch
        label_fake = torch.full((self.batch_size,), 1, dtype=torch.float, device=self.device)  # Label 1 cho fake
        output_fake = self.model_d(self.image_1, self.fake_image_2.detach()).view(-1)
        errD_fake = self.contrastive_loss(output_fake, label_fake)

        errD = self.triplet_loss(output_real, output_fake) + errD_real + errD_fake
        errD.backward()
        self.optimizer_d.step()
        return errD

    def train_g(self):
        # -----------------
        #  Train Generator
        # -----------------

        self.optimizer_g.zero_grad()
        label_g = torch.full((self.batch_size,), 0, dtype=torch.float, device=self.device)
        output_g = self.model_d(self.image_1, self.fake_image_2).view(-1)
        errG = self.contrastive_loss(output_g, label_g) + self.mse_loss(self.image_1, self.fake_image_2)
        errG.backward()
        self.optimizer_g.step()
        return errG

    def test(self, epoch, tsne, flag, roc):
        self.load_model(epoch)
        print("Testing ...")
        return self.evaluate(epoch, self.model_d, tsne, flag, roc)

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
        accuracy, threshold = self.accuracy(test_distances, test_labels)

        test_distances_cpu = test_distances_meter.get_val_numpy()
        test_labels_cpu = test_labels_meter.get_val_numpy()
        predict_y = (test_distances_cpu < threshold.numpy()).astype(int)
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
        print("SAVING MODEL AT", str(epoch), " ...")
        to_save = {
            'model_d': self.model_d.state_dict(),
            'model_g': self.model_g.state_dict(),
            'optim_d': self.optimizer_d.state_dict(),
            'optim_g': self.optimizer_g.state_dict(),
        }

        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(to_save, os.path.join(self.model_dir, f'model_fold{self.fold}_{str(epoch)}.pth'))

    def load_model(self, epoch):
        print("LOADING MODEL FROM", str(epoch), " ...")
        checkpoint = torch.load(os.path.join(self.model_dir,
                                             f'model_fold{self.fold}_{str(epoch)}.pth'), weights_only=True)

        # Load model state
        self.model_d.load_state_dict(checkpoint['model_d'])
        self.model_g.load_state_dict(checkpoint['model_g'])

        # Load optimizer state
        if 'optim_d' in checkpoint and self.optimizer_d is not None:
            self.optimizer_d.load_state_dict(checkpoint['optim_d'])

        if 'optim_g' in checkpoint and self.optimizer_g is not None:
            self.optimizer_g.load_state_dict(checkpoint['optim_g'])

        print("MODEL LOADED SUCCESSFULLY")
