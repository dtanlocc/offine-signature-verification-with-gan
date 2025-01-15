import json
import os

from torchvision import transforms

from config.defaults import get_cfg_defaults
from trainer.SignGanTrain import SignGanTrainer
from utils.create_dataloader import create_train_test_loaders
from utils.helpers import test_generator

if '__main__' == __name__:
    opt = get_cfg_defaults()
    opt.freeze()

    csv_files = ['D:/LVTN/data/CEDAR/data_fold_1.csv',
                 'D:/LVTN/data/CEDAR/data_fold_2.csv',
                 'D:/LVTN/data/CEDAR/data_fold_3.csv',
                 'D:/LVTN/data/CEDAR/data_fold_4.csv',
                 'D:/LVTN/data/CEDAR/data_fold_5.csv']
    path_root = 'D:/LVTN/data/CEDAR/CEDAR'

    transform = transforms.Compose([
        transforms.Resize((155, 220)),
        transforms.CenterCrop((155, 220)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    results = {
        'GAN': {f'fold_{i}': {'far': [], 'frr': [], 'acc': [], 'loss_d': [], 'loss_g': []} for i in range(5)}
    }

    for idx_fold_test in range(5):
        print('_______________________________________________________________________________')
        print(f"FOLD {idx_fold_test} START!!!")
        train_loader, test_loader = create_train_test_loaders(csv_files,
                                                              path_root,
                                                              transform=transform,
                                                              batch_size=opt.BATCH_SIZE,
                                                              idx_fold_test=idx_fold_test)
        trainer_SignGan = SignGanTrainer(opt, fold=str(idx_fold_test), train_dataloader=train_loader,
                                         test_dataloader=test_loader)
        trainer_SignGan.train()
        # trainer_SignNet = SignNetTrainer(opt, fold=str(idx_fold_test), train_dataloader=train_loader, test_dataloader=test_loader)
        # trainer_SignNet.train()

        results['GAN'][f'fold_{idx_fold_test}'] = trainer_SignGan.results['GAN'][f'fold_{idx_fold_test}']
        # results['SIGNET'][f'fold_{idx_fold_test}'] = trainer_SignNet.results['SIGNET'][f'fold_{idx_fold_test}']
        test_generator(trainer_SignGan.img_list, save_folder=f"{opt.SAVE_DIR_GAN}/fold_{idx_fold_test}")

    # Save all results after training all folds
    final_results_path = os.path.join(opt.SAVE_DIR_GAN, 'final_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Final results saved to {final_results_path}")
