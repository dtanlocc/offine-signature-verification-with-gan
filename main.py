from torchvision import transforms
from utils.transform import Resize, RandomCrop, StandardizeImage
from utils.create_dataloader import create_train_test_loaders
from config.defaults import get_cfg_defaults
from trainer.trainer import clf_signatureTrainer


if '__main__' == __name__:
    opt = get_cfg_defaults()
    opt.freeze()
    resize_transform = Resize(output_size=(64, 64))
    random_crop_transform = RandomCrop(output_size=(64, 64))

    csv_files = ['D:/LVTN/data/CEDAR/data_fold_1.csv',
                 'D:/LVTN/data/CEDAR/data_fold_2.csv',
                 'D:/LVTN/data/CEDAR/data_fold_3.csv',
                 'D:/LVTN/data/CEDAR/data_fold_4.csv',
                 'D:/LVTN/data/CEDAR/data_fold_5.csv']
    path_root = 'D:/LVTN/data/CEDAR/CEDAR'
    transform = transforms.Compose([resize_transform,
                                    random_crop_transform,
                                    transforms.ToTensor(),
                                    StandardizeImage()])
    results = {
        'GAN': {'fold_0': {'far': [], 'frr': [], 'acc': [], 'optimal_threshold': [], 'loss_d': [], 'loss_g': []},
                'fold_1': {'far': [], 'frr': [], 'acc': [], 'optimal_threshold': [], 'loss_d': [], 'loss_g': []},
                'fold_2': {'far': [], 'frr': [], 'acc': [], 'optimal_threshold': [], 'loss_d': [], 'loss_g': []},
                'fold_3': {'far': [], 'frr': [], 'acc': [], 'optimal_threshold': [], 'loss_d': [], 'loss_g': []},
                'fold_4': {'far': [], 'frr': [], 'acc': [], 'optimal_threshold': [], 'loss_d': [], 'loss_g': []}, }

    }

    for idx_fold_test in range(5):
        print('_______________________________________________________________________________')
        print(f"FOLD {idx_fold_test} START!!!")
        train_loader, test_loader, num_users = create_train_test_loaders(csv_files,
                                                                         path_root,
                                                                         transform=transform,
                                                                         batch_size=opt.BATCH_SIZE,
                                                                         idx_fold_test=idx_fold_test)

        trainer = clf_signatureTrainer(opt, fold=str(idx_fold_test), train_dataloader=train_loader, test_dataloader=test_loader)
        trainer.train()






