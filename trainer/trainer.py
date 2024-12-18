import torch
import torch.optim as optim
from utils.helpers import load_model, plot_tsne, plot_confusion_matrix, visualize_metrics_seaborn
from utils.discriminator_loss import triplet_loss, contrastive_loss
import torchvision.utils as vutils
from sklearn.neighbors import KNeighborsClassifier
from utils.meters import CatMeter, AverageMeter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os


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
        self.fixed_noise = torch.randn(64, opt.NZ, 1, 1, device=self.device)
        self.save_dir = opt.SAVE_DIR
        self.model_dir = opt.MODEL_DIR
        self.accuracies = []
        self.train_g_losses = []
        self.train_d_losses = []

        self._init_model()
        self._init_criterion()
        self._init_optimizer()

    def _init_model(self):
        self.model_d, self.model_g = load_model(self.opt, self.device)

    def _init_optimizer(self):
        self.optimizer_g = optim.Adam(self.model_g.parameters(), lr=self.opt.LEARNING_RATE, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.model_d.parameters(), lr=self.opt.LEARNING_RATE, betas=(0.5, 0.999))

    def _init_criterion(self):
        self.triplet_loss = triplet_loss
        self.contrastive_loss = contrastive_loss

    def train(self):
        best_accuracy = 0.0
        best_epoch = 0

        print('START TRAINING.....')
        for epoch in range(self.epochs):
            self.model_d.train()
            self.model_g.train()
            train_dis_loss, train_gen_loss = self._train_one_epoch(epoch)
            self.train_g_losses.append(train_dis_loss)
            self.train_d_losses.append(train_gen_loss)

            if (self.eval_step > 0) and (epoch + 1) % self.eval_step == 0 and epoch + 1 == self.epochs:
                accuracy, _, _ = self.evaluate(epoch, tsne=False)
                self.accuracies.append(accuracy)
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch + 1
                    self.save_model(epoch + 1)
        accuracy, cm, classification_report_dict = self.evaluate(best_epoch, tsne=True)
        visualize_metrics_seaborn(cm, self.save_dir)
        plot_confusion_matrix(cm, self.save_dir)
        print(f'Best Accuracy: {best_accuracy:.4f}, epoch: {best_epoch}')

    def _train_one_epoch(self, epoch):
        self.d_loss_meter = AverageMeter()
        self.g_loss_meter = AverageMeter()

        for i, sample in enumerate(self.train_dataloader):
            # Dữ liệu đầu vào
            self.user_id = sample['user'].to(self.device)
            self.image_1 = sample['image_1'].to(self.device)
            self.image_2 = sample['image_2'].to(self.device)
            self.label = sample['label'].to(self.device).float()
            self.fake_labels = torch.zeros(self.image_1.size(0)).view(-1, 1).to(self.device)
            self.real_labels = torch.ones(self.image_1.size(0)).view(-1, 1).to(self.device)

            z = torch.randn(self.image_1.size(0), 100,1,1, device=self.device)

            self.fake_image_2 = self.model_g(z, self.user_id)

            # Train Discriminator và lưu loss
            d_loss = self.train_d()
            self.d_loss_meter.update(d_loss, n=self.image_1.size(0))

            # Train Generator và lưu loss
            g_loss = self.train_g()
            self.g_loss_meter.update(g_loss, n=self.image_1.size(0))

            del self.user_id
            del self.image_1
            del self.fake_image_2
            del self.image_2

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = self.model_g(self.fixed_noise, torch.full((64,), 44, dtype=torch.int).to(self.device)).detach().cpu()
        self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # Ghi nhận loss mean
        d_loss_avg, _ = self.d_loss_meter.value()
        g_loss_avg, _ = self.g_loss_meter.value()

        # Ghi nhận loss mean
        print(f"EPOCH {epoch + 1}:\tTrain dis loss: {d_loss_avg:.4f}\tTrain gen loss: {g_loss_avg:.4f}")

        return d_loss_avg, g_loss_avg

    def train_d(self):
        # Zero gradients for Discriminator
        self.optimizer_d.zero_grad()

        # Cosine similarity for real images
        real_dist = self.model_d(self.image_1, self.image_2, self.user_id)

        # Cosine similarity for fake images
        fake_dist = self.model_d(self.image_1, self.fake_image_2.detach(), self.user_id)

        # Compute the loss for real and fake images
        d_loss_real = self.contrastive_loss(real_dist, self.label.view(-1, 1).to(self.device).float())
        d_loss_fake = self.contrastive_loss(fake_dist, self.fake_labels)

        # Triplet loss between real and fake images
        triplet_loss_value = self.triplet_loss(real_dist, fake_dist)

        # Total Discriminator loss
        d_loss = (d_loss_real + d_loss_fake) + triplet_loss_value

        # Backpropagation and update the Discriminator
        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def train_g(self):
        self.optimizer_g.zero_grad()

        # Get the discriminator's output for fake images
        features_real = self.model_d.forward_once(self.image_1, self.user_id)
        features_fake = self.model_d.forward_once(self.fake_image_2, self.user_id)

        # Calculate Generator loss
        loss_fr = torch.mean((features_fake - features_real) ** 2)

        # Adversarial Loss
        dist_gen = self.model_d(self.image_1, self.fake_image_2, self.user_id)
        loss_adv = contrastive_loss(dist_gen, self.real_labels)

        # Tổng Loss của Generator
        g_loss = loss_adv + loss_fr

        # Backpropagation and update the Generator
        g_loss.backward()
        self.optimizer_g.step()

        return g_loss.item()

    def _update_values(self, epoch):
        pass

    def test(self, epoch, tsne):
        self.load_model(epoch)
        print("Testing ...")
        self.evaluate(epoch, tsne)

    def evaluate(self, epoch, tsne=False):
        print("START EVALUATION....")
        self.model_d.eval()

        train_distances_meter, train_labels_meter = CatMeter(), CatMeter()
        test_distances_meter, test_labels_meter = CatMeter(), CatMeter()
        print("Calculating training distance ...")
        with torch.no_grad():
            for i, sample in enumerate(self.train_dataloader):
                user_id = sample['user'].to(self.device)
                image_1 = sample['image_1'].to(self.device)
                image_2 = sample['image_2'].to(self.device)
                label = sample['label'].to(self.device).float()

                # Distance image_1, image_2 train_dataset
                dist = self.model_d(image_1, image_2, user_id)
                train_distances_meter.update(dist)
                train_labels_meter.update(label)

        with torch.no_grad():
            for i, sample in enumerate(self.test_dataloader):
                user_id = sample['user'].to(self.device)
                image_1 = sample['image_1'].to(self.device)
                image_2 = sample['image_2'].to(self.device)
                label = sample['label'].to(self.device).float()

                # Distance image_1, image_2 test_dataset
                dist = self.model_d(image_1, image_2, user_id)
                test_distances_meter.update(dist)
                test_labels_meter.update(label)

        # Convert kết quả sang numpy
        train_distances = train_distances_meter.get_val_numpy()
        train_labels = train_labels_meter.get_val_numpy()

        test_distances = test_distances_meter.get_val_numpy()
        test_labels = test_labels_meter.get_val_numpy()

        # Chuyển đổi thành mảng 2D
        train_distances = train_distances.reshape(-1, 1)
        test_distances = test_distances.reshape(-1, 1)

        # KNN Classifier
        print("Evaluating with KNN....")
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(train_distances, train_labels)
        predict_y = knn.predict(test_distances)

        accuracy = accuracy_score(test_labels, predict_y)
        print(f"Accuracy at epoch {epoch + 1}: {accuracy:.4f}")

        # Tính confusion matrix
        cm = confusion_matrix(test_labels, predict_y)
        print("Confusion Matrix:")
        print(cm)

        class_report = classification_report(test_labels, predict_y, output_dict=True)
        print("Classification Report:")
        print(class_report)

        if tsne:
            plot_tsne(train_distances, train_labels, self.save_dir, epoch)

        del knn
        del train_labels
        del train_distances
        del test_labels
        del test_distances

        return accuracy, cm, class_report

    def save_model(self, epoch):
        print("SAVING MODEL AT", str(epoch), " ...")

        torch.save(self.model_g.state_dict(), os.path.join(self.model_dir, f"generator_fold{self.fold}_{str(epoch)}.pth"))
        torch.save(self.model_d.state_dict(), os.path.join(self.model_dir,
                                                           f'discriminator_fold{self.fold}_{str(epoch)}.pth'))

    def load_model(self, epoch):
        print("LOADING MODEL")

        self.model_g.load_state_dict(torch.load(os.path.join(self.model_dir,
                                                             f'generator_fold{self.fold}_{str(epoch)}.pth')))
        self.model_d.load_state_dict(torch.load(os.path.join(self.model_dir,
                                                             f'discriminator_fold{self.fold}_{str(epoch)}.pth')))
