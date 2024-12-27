import torch
import torch.optim as optim
from utils.helpers import load_model, plot_confusion_matrix, visualize_metrics_seaborn, calculator_metric
from utils.discriminator_loss import triplet_loss, contrastive_loss
import torchvision.utils as vutils
import json
from utils.meters import CatMeter, AverageMeter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from tqdm.notebook import tqdm
import torch.nn.functional as F


class clf_signatureTrainer:
    def __init__(self, opt, fold, train_dataloader, test_dataloader):
        torch.manual_seed(opt.SEED)
        self.opt = opt
        if opt.USE_GPU:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.batch_size = opt.BATCH_SIZE
        self.epochs = opt.EPOCHS
        self.eval_step = opt.EVAL_STEP
        self.fold = fold
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.img_list = []
        self.fixed_noise = torch.randn(opt.BATCH_SIZE, opt.NZ, device=self.device)
        self.save_dir = opt.SAVE_DIR
        self.model_dir = opt.MODEL_DIR
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

    def _init_model(self):
        self.model_d, self.model_g = load_model(self.opt, self.device)

    def _init_optimizer(self):
        self.optimizer_g = optim.Adam(self.model_g.parameters(), lr=self.opt.LEARNING_RATE, betas=(0.5, 0.999))
        self.optimizer_d = optim.RMSprop(self.model_d.parameters(), lr=self.opt.LEARNING_RATE, eps=1e-8,
                                         weight_decay=5e-4, momentum=0.9)

    def _init_criterion(self):
        self.triplet_loss = triplet_loss
        self.contrastive_loss = contrastive_loss

    def _init_scheduler(self):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer_d, 5, 0.1)

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

            self.results['GAN'][f'fold_{self.fold}']['loss_d'].append(train_dis_loss)
            self.results['GAN'][f'fold_{self.fold}']['loss_g'].append(train_gen_loss)
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

    def save_results(self, results):
        os.makedirs(self.save_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        save_path = os.path.join(self.save_dir, f'results_fold{self.fold}.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {save_path}")

    def _train_one_epoch(self, epoch):
        self.d_loss_meter = AverageMeter()
        self.g_loss_meter = AverageMeter()

        progress_bar = tqdm(self.train_dataloader)
        for i, sample in enumerate(progress_bar):
            # Dữ liệu đầu vào
            self.image_1 = sample['image_1'].to(self.device)
            self.image_2 = sample['image_2'].to(self.device)
            self.label = sample['label'].to(self.device).float()
            self.real_labels = torch.zeros(self.image_1.size(0)).view(-1, 1).to(self.device)
            self.fake_labels = torch.ones(self.image_1.size(0)).view(-1, 1).to(self.device)

            z = torch.randn(self.image_1.size(0), self.opt.NZ, 1, 1, device=self.device)

            self.fake_image_2 = self.model_g(z)

            # Train Discriminator và lưu loss
            d_loss = self.train_d()
            self.d_loss_meter.update(d_loss)

            # Train Generator và lưu loss
            g_loss = self.train_g()
            self.g_loss_meter.update(g_loss)

            progress_bar.set_description(
                f"EPOCH [{epoch}/{self.epochs}]:Train dis loss: {d_loss:.4f}\tTrain gen loss: {g_loss:.4f}")

            del self.image_1, self.fake_image_2, self.image_2, z
            torch.cuda.empty_cache()

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = self.model_g(self.fixed_noise).to(self.device).detach().cpu()
        self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # Ghi nhận loss mean
        d_loss_avg, _ = self.d_loss_meter.value()
        g_loss_avg, _ = self.g_loss_meter.value()

        accuracy, cm, classification_report_dict = self.evaluate(epoch, tsne=False, flag=False, roc=True)
        self.scheduler.step()

        # Ghi nhận loss mean
        print(
            f"EPOCH {epoch}:\tTrain dis loss: {d_loss_avg:.4f}\tTrain gen loss: {g_loss_avg:.4f}\tAccuracy: {accuracy:.4f}")
        log_file = os.path.join(self.save_dir, f'train_log_fold{self.fold}.txt')
        with open(log_file, 'a') as log:
            log.write(
                f"Epoch {epoch}:\tTrain dis loss: {d_loss_avg:.4f}\tTrain gen loss: {g_loss_avg:.4f}\tAccuracy: {accuracy:.4f}\n")
        return d_loss_avg, g_loss_avg, accuracy, cm, classification_report_dict

    def train_d(self):
        # Zero gradients for Discriminator
        self.optimizer_d.zero_grad()

        # feature_image from discriminator
        image_features_1, image_features_2 = self.model_d(self.image_1, self.image_2)

        # Calculator euclid
        euclidean_distance_real = F.pairwise_distance(image_features_1, image_features_2, p=2)

        # Cosine similarity for fake images
        image_features_1, image_features_2_fake = self.model_d(self.image_1, self.fake_image_2.detach())
        euclidean_distance_fake = F.pairwise_distance(image_features_1, image_features_2_fake, p=2)

        # Compute the loss for real and fake images
        d_loss_real = self.contrastive_loss(euclidean_distance_real, self.label.view(-1, 1).to(self.device).float())
        d_loss_fake = self.contrastive_loss(euclidean_distance_fake, self.fake_labels)

        # Triplet loss between real and fake images
        triplet_loss_value = self.triplet_loss(euclidean_distance_real, euclidean_distance_fake)

        # Total Discriminator loss
        d_loss = (d_loss_real + d_loss_fake) + triplet_loss_value

        # Backpropagation and update the Discriminator
        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def train_g(self):
        self.optimizer_g.zero_grad()

        # Get the discriminator's output for fake images
        features_real, features_fake = self.model_d(self.image_1, self.fake_image_2)

        # Calculate Generator loss
        loss_fr = torch.mean((features_fake - features_real) ** 2)

        # # Adversarial Loss
        dist_gen = F.pairwise_distance(features_real, features_fake, p=2)
        loss_adv = contrastive_loss(dist_gen, self.real_labels)

        # Tổng Loss của Generator
        g_loss = loss_fr + loss_adv

        # Backpropagation and update the Generator
        g_loss.backward()
        self.optimizer_g.step()
        del features_real, features_fake
        torch.cuda.empty_cache()

        return g_loss.item()

    def _update_values(self, epoch):
        pass

    def test(self, epoch, tsne, flag, roc):
        self.load_model(epoch)
        print("Testing ...")
        return self.evaluate(epoch, tsne, flag, roc)

    def accuracy(self, distances, y, step=0.0001):
        min_threshold_d = min(distances)
        max_threshold_d = max(distances)

        # step = (max_threshold_d-min_threshold_d)/1000
        max_acc = 0
        same_id = (y == 1)
        d_optimal = float('inf')

        # Duyệt qua các ngưỡng
        thresholds = torch.arange(min_threshold_d, max_threshold_d + step, step)
        for threshold_d in thresholds:
            true_positive = (distances <= threshold_d) & same_id
            true_negative = (distances > threshold_d) & (~same_id)

            # Tính tỷ lệ đúng
            true_positive_rate = true_positive.sum().float() / same_id.sum().float()
            true_negative_rate = true_negative.sum().float() / (~same_id).sum().float()

            # Tính accuracy
            acc = 0.5 * (true_negative_rate + true_positive_rate)
            if acc > max_acc:
                max_acc = acc
                d_optimal = threshold_d

        return max_acc, d_optimal

    def evaluate(self, epoch, tsne=False, flag=False, roc=True):
        # print("START EVALUATION....")
        self.model_d.eval()

        test_distances_meter, test_labels_meter = CatMeter(), CatMeter()
        print("Calculating training distance ...")

        with torch.no_grad():
            for i, sample in enumerate(self.test_dataloader):
                image_1 = sample['image_1'].to(self.device)
                image_2 = sample['image_2'].to(self.device)
                label = sample['label'].to(self.device).float()

                feature_1, feature_2 = self.model_d(image_1, image_2)
                dist = F.pairwise_distance(feature_1, feature_2, p=2)
                test_distances_meter.update(dist)
                test_labels_meter.update(label)

                del feature_1, feature_2
                torch.cuda.empty_cache()

        test_distances = test_distances_meter.get_val()
        test_labels = test_labels_meter.get_val()

        accuracy, thresold = self.accuracy(test_distances, test_labels)

        test_distances_cpu = test_distances_meter.get_val_numpy()
        test_labels_cpu = test_labels_meter.get_val_numpy()
        predict_y = (test_distances_cpu < thresold.numpy()).astype(int)

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
            pass

        del test_labels
        del test_distances

        return accuracy, cm, class_report

    def save_model(self, epoch):
        print("SAVING MODEL AT", str(epoch), " ...")
        to_save = {
            'model': self.model_d.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'optim': self.optimizer_d.state_dict(),
        }

        torch.save(to_save, os.path.join(self.model_dir,
                                         f'discriminator_fold{self.fold}_{str(epoch)}.pth'))
        torch.save(self.model_g.state_dict(),
                   os.path.join(self.model_dir, f"generator_fold{self.fold}_{str(epoch)}.pth"))

    def load_model(self, epoch):
        print("LOADING MODEL FROM", str(epoch), " ...")
        checkpoint = torch.load(os.path.join(self.model_dir,
                                             f'discriminator_fold{self.fold}_{str(epoch)}.pth'))

        # Load model state
        self.model_d.load_state_dict(checkpoint['model'])

        # Load optimizer state
        if 'optim' in checkpoint and self.optimizer_d is not None:
            self.optimizer_d.load_state_dict(checkpoint['optim'])

        # Load scheduler state
        if 'scheduler' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.model_g.load_state_dict(torch.load(os.path.join(self.model_dir,
                                                             f'generator_fold{self.fold}_{str(epoch)}.pth')))

        print("MODEL LOADED SUCCESSFULLY")