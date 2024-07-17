###############################################################################
# Imports
###############################################################################

import numpy as np
import torch
from pathlib import Path
import argparse
import os

from models.EEGModels import EEGNet, EEGNet_SSVEP, EEGNet_ChanRed, EEGNet_WT, TCNet, TCNet_EMD
from preprocess.preprocess_SEED import SEEDDataset
from tools import train_f, test_f, xDawnRG, classification_accuracy, draw_loss, MODEL_CHOICES
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader, SubsetRandomSampler

import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.train import RunConfig
import ray



def eval_SEED(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if device.type != "cuda":
        raise Exception("CUDA not available. Please select a GPU device.")
    else:
        print('Using device:', device)
    properties = torch.cuda.get_device_properties(device)
    n_cpu = os.cpu_count()
    n_gpu = torch.cuda.device_count()
    accelerator = properties.name.split()[1]

    selected_model = args.model
    is_ok = selected_model != 'TCNet' and device.type != 'mps' #TODO: understand why TCNet doesn't work with mixed precision (probably overflows)


    def train_eval_SEED():

        ###############################################################################
        # Hyperparameters
        ###############################################################################

        best_start = 1
        best_sample = 200
        best_std = True
        subjects = [[i] for i in range(15)]
        sessions = None

        epochs_dep = 200
        epochs_ind = 20
        random_seed= 42

        best_lr = 0.001
        best_batch_size = 128
        best_F1 = 64
        best_D = 8
        best_F2 = 64
        best_kernLength = 15 # perhaps go back to 100 for f_min = 2Hz
        best_dropout = .1
        best_norm_rate = .25
        best_nr = 1.
        best_tfr = None

        names = ['Negative', 'Neutral', 'Positive']
        selected_emotion = 'happiness(SEED)'

        n_components = 2  # pick some components for xDawnRG
        nb_classes = len(names)
        chans = 62
        splits = KFold(n_splits=10, shuffle=True, random_state=random_seed)
        best_innerChans = 24
        best_shifted = True

        cur_dir = Path(__file__).resolve().parent
        figs_path = str(cur_dir) + '/figs/'
        sets_path = str(cur_dir) + '/sets/'
        save = False

        np.random.seed(random_seed)
        dependent = True
        independent = False


        ###############################################################################
        # Subject-dependent classification
        ###############################################################################

        if dependent:
            preds = []
            Y_test = []
            accs = []
            for subject in subjects:
                preds_sub = []
                Y_test_sub = []

                info_str = 'SEED_' + f'_subject({subject})_samples({best_sample})_start({best_start})_'


                ###############################################################################
                # Data loading
                ###############################################################################

                dataset = SEEDDataset(sets_path+info_str, subjects=subject, sessions=None, samples=best_sample, start=best_start, save=save,
                                    tfr=best_tfr, std=best_std, n_jobs=n_cpu)
                dataset_size = len(dataset)

                for i, (train_idx, val_idx) in enumerate(splits.split(list(range(dataset_size)))):
                    print("Fold no.{}:".format(i + 1))
                    train_sampler = SubsetRandomSampler(train_idx)
                    valid_sampler = SubsetRandomSampler(val_idx)
                    train_loader = DataLoader(dataset, batch_size=best_batch_size, sampler=train_sampler, pin_memory=True)
                    train_loader = ray.train.torch.prepare_data_loader(train_loader)
                    valid_loader = DataLoader(dataset, batch_size=best_batch_size, sampler=valid_sampler, pin_memory=True)
                    print(len(train_idx), 'train samples')
                    print(len(val_idx), 'test samples')


                    ###############################################################################
                    # Model configurations
                    ###############################################################################

                    if selected_model == 'EEGNet':
                        model = EEGNet_WT(nb_classes=nb_classes, Chans=chans, InnerChans=best_innerChans, Samples=best_sample, dropoutRate=best_dropout,
                                            kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2,
                                            norm_rate=best_norm_rate, nr=best_nr, dropoutType='Dropout', nb_freqs=best_tfr+1).to(memory_format=torch.channels_last)
                    elif selected_model == 'TCNet':
                        model = TCNet_EMD(nb_classes, chans, nb_freqs=best_tfr+1, shifted=best_shifted, kern_emd=best_kernLength)
                    else:
                        raise ValueError('Invalid model selected')
                    model = ray.train.torch.prepare_model(model)

                    loss_fn = torch.nn.CrossEntropyLoss().to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
                    scaler = torch.cuda.amp.GradScaler(enabled=is_ok)

                    # torch.backends.cudnn.benchmark = True


                    ###############################################################################
                    # Train and test
                    ###############################################################################

                    losses_train = []
                    losses_test = []
                    for epoch in range(epochs_dep):
                        if ray.train.get_context().get_world_size() > 1:
                            train_loader.sampler.set_epoch(epoch)

                        loss = train_f(model, train_loader, optimizer, loss_fn, scaler, device, is_ok, ddp=True)
                        losses_train.append(loss)
                        acc, loss_test = test_f(model, valid_loader, loss_fn, device, is_ok)
                        losses_test.append(loss_test)
                        if epoch % 20 == 0:
                            print(f"Epoch {epoch}: Train loss: {loss}, Test accuracy: {acc}, Test loss: {loss_test}")
                    draw_loss(losses_train, losses_test, figs_path, selected_emotion, str(subject))

                    with torch.no_grad():
                        for X_batch, Y_batch in valid_loader:
                            if selected_model != 'TCNet':
                                X_batch = X_batch.to(device=device, memory_format=torch.channels_last)
                            else:
                                X_batch = X_batch.to(device=device)
                            if is_ok:
                                with torch.autocast(device_type=device.type, dtype=torch.float16):
                                    y_pred = model(X_batch)
                            else:
                                y_pred = model(X_batch)
                            _, predicted = torch.max(y_pred.data, 1)
                            preds.append(predicted.cpu().numpy())
                            preds_sub.append(predicted.cpu().numpy())
                            target = Y_batch
                            Y_test.append(target.numpy())
                            Y_test_sub.append(target.numpy())

                acc = np.mean(np.concatenate(preds_sub) == np.concatenate(Y_test_sub))
                accs.append(acc)
            classification_accuracy(np.concatenate(preds), np.concatenate(Y_test), names, figs_path, selected_emotion, 'dependent', accs)


        ###############################################################################
        # Subject-independent classification
        ###############################################################################

        if independent:
            preds = []
            Y_test = []
            accs = []
            for subject in subjects:

                info_str_test = 'SEED_' + selected_emotion + f'_subject({subject})_samples({best_sample})_start({best_start})_'


                ###############################################################################
                # Data loading
                ###############################################################################

                subjects_train = [i for i in range(23) if i != subject[0]]
                info_str_train = 'SEED_' + selected_emotion + f'_subject({subjects_train})_samples({best_sample})_start({best_start})_'
                subjects_test = subject
                dataset_train = SEEDDataset(sets_path+info_str_train, subjects=subjects_train, sessions=sessions, samples=best_sample, start=best_start,
                                            save=save, tfr=best_tfr, std=best_std, n_jobs=n_cpu)
                dataset_test = SEEDDataset(sets_path+info_str_test, subjects=subjects_test, sessions=sessions, samples=best_sample, start=best_start,
                                        save=save, tfr=best_tfr, std=best_std, n_jobs=n_cpu)
                dataset_train_size = len(dataset_train)
                dataset_test_size = len(dataset_test)

                train_indices = list(range(dataset_train_size))
                test_indices = list(range(dataset_test_size))
                np.random.shuffle(train_indices)
                np.random.shuffle(test_indices)

                # Creating data samplers and loaders:
                train_sampler = SubsetRandomSampler(train_indices)
                test_sampler = SubsetRandomSampler(test_indices)
                train_loader = DataLoader(dataset_train, batch_size=best_batch_size, sampler=train_sampler, pin_memory=True)
                train_loader = ray.train.torch.prepare_data_loader(train_loader)
                test_loader = DataLoader(dataset_test, batch_size=best_batch_size, sampler=test_sampler, pin_memory=True)
                print(len(train_indices), 'train samples')
                print(len(test_indices), 'test samples')


                ###############################################################################
                # Model configurations
                ###############################################################################

                if selected_model == 'EEGNet':
                    model = EEGNet_WT(nb_classes=nb_classes, Chans=chans, InnerChans=best_innerChans, Samples=best_sample,
                                      dropoutRate=best_dropout, kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2,
                                      norm_rate=best_norm_rate, nr=best_nr, dropoutType='Dropout', nb_freqs=best_tfr+1).to(memory_format=torch.channels_last)
                elif selected_model == 'TCNet':
                    model = TCNet_EMD(nb_classes, chans, nb_freqs=best_tfr+1, shifted=best_shifted, kern_emd=best_kernLength)
                else:
                    raise ValueError('Invalid model selected')
                model = ray.train.torch.prepare_model(model)

                loss_fn = torch.nn.CrossEntropyLoss().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
                scaler = torch.cuda.amp.GradScaler(enabled=is_ok)

                # torch.backends.cudnn.benchmark = True


                ###############################################################################
                # Train and test
                ###############################################################################

                for epoch in range(epochs_ind):
                    loss = train_f(model, train_loader, optimizer, loss_fn, scaler, device, is_ok)
                    acc, loss_test = test_f(model, test_loader, loss_fn, device, is_ok)
                    if epoch % 1 == 0:
                        print(f"Epoch {epoch}: Train loss: {loss}, Test accuracy: {acc}, Test loss: {loss_test}")

                with torch.no_grad():
                    for X_batch, Y_batch in test_loader:
                        if selected_model != 'TCNet':
                            X_batch = X_batch.to(device=device, memory_format=torch.channels_last)
                        else:
                            X_batch = X_batch.to(device=device)
                        if is_ok:
                            with torch.autocast(device_type=device.type, dtype=torch.float16):
                                y_pred = model(X_batch)
                        else:
                            y_pred = model(X_batch)
                        _, predicted = torch.max(y_pred.data, 1)
                        preds.append(predicted.cpu().numpy())
                        target = Y_batch
                        Y_test.append(target.numpy())

                acc = np.mean(np.concatenate(preds) == np.concatenate(Y_test))
                accs.append(acc)
            classification_accuracy(np.concatenate(preds), np.concatenate(Y_test), names, figs_path, selected_emotion, 'independent', accs)

    ray.init(num_cpus=n_cpu, num_gpus=n_gpu)
    trainer = TorchTrainer(
        train_eval_SEED,
        scaling_config=ScalingConfig(num_workers=n_gpu,
                                     use_gpu=True,
                                     resources_per_worker={"CPU": (n_cpu-1)/n_gpu, "GPU": 1.0},
                                     accelerator_type=accelerator),
        run_config=RunConfig(verbose=0)
    )

    results = trainer.fit()


    ###############################################################################
    # Statistical benchmark analysis
    ###############################################################################

    # xDawnRG(dataset, n_components, train_indices, test_indices, chans, samples, names, figs_path, info_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates the performance on the SEED dataset.")
    parser.add_argument('model', choices=MODEL_CHOICES)
    args = parser.parse_args()
    n_gpus = torch.cuda.device_count()

    if n_gpus >= 2:
        print('Coming soon...')

    else:
        eval_SEED(args)
