###############################################################################
# Imports
###############################################################################

import numpy as np
import torch
from pathlib import Path

from models.EEGModels import EEGNet, EEGNet_SSVEP
from preprocess.preprocess_DREAMER import DREAMERDataset
from tools import train_f, test_f, xDawnRG, classification_accuracy

from torch.utils.data import DataLoader, SubsetRandomSampler


###############################################################################
# Hyperparameters
###############################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

best_highcut = None
best_lowcut = .5
best_order = 3
best_type = 'butter'
best_start = 1
best_sample = 128
subjects = [[i] for i in range(23)]
sessions = [[i] for i in range(18)]

epochs_dep_mix = 800
epochs_dep_ind = 800
epochs_ind = 100
random_seed= 42
test_split = .25

best_lr = 0.001
best_batch_size = 128
best_F1 = 64
best_D = 8
best_F2 = 64
best_kernLength = 16 # maybe go back to 64 because now f_min = 8Hz
best_dropout = .1

selected_emotion = 'valence'
class_weights = torch.tensor([1., 1., 1., 1., 1.]).to(device)
names = ['1', '2', '3', '4', '5']

n_components = 2  # pick some components for xDawnRG
nb_classes = 5
chans = 14

cur_dir = Path(__file__).resolve().parent
figs_path = str(cur_dir) + '/figs/'
sets_path = str(cur_dir) + '/sets/'
models_path = str(cur_dir) + '/tmp/'
save = False

np.random.seed(random_seed)
dep_mix = False
dep_ind = True
independent = True


###############################################################################
# Subject-dependent mixed sessions classification
###############################################################################

if dep_mix:
    preds = []
    Y_test = []
    for subject in subjects:

        info_str = 'DREAMER_' + selected_emotion + f'_subject({subject})_session({sessions})_filtered({best_lowcut}, {best_highcut}, {best_order})_samples({best_sample})_start({best_start})_'


        ###############################################################################
        # Data loading
        ###############################################################################

        dataset = DREAMERDataset(sets_path+info_str, selected_emotion, subjects=subject, sessions=None, samples=best_sample, start=best_start,
                                lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save)
        dataset_size = len(dataset)

        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split_test = int(np.floor(test_split * dataset_size))
        train_indices, test_indices = indices[split_test:], indices[:split_test]

        # Creating data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(dataset, batch_size=best_batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=best_batch_size, sampler=test_sampler)

        print(len(train_indices), 'train samples')
        print(len(test_indices), 'test samples')


        ###############################################################################
        # Model configurations
        ###############################################################################

        model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=best_sample, 
                    dropoutRate=best_dropout, kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2, dropoutType='Dropout').to(device)

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)


        ###############################################################################
        # Train and test
        ###############################################################################

        for epoch in range(epochs_dep_mix):
            loss = train_f(model, train_loader, optimizer, loss_fn, device)
            acc, loss_test = test_f(model, test_loader, loss_fn, device)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train loss: {loss}, Test accuracy: {acc}, Test loss: {loss_test}")

        for batch_index, (X_batch, Y_batch) in enumerate(test_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            y_pred = model(X_batch)
            _, predicted = torch.max(y_pred.data, 1)
            preds.append(predicted.cpu().numpy())
            _, target = torch.max(Y_batch, 1)
            Y_test.append(target.cpu().numpy())

    classification_accuracy(np.concatenate(preds), np.concatenate(Y_test), names, figs_path, selected_emotion, 'dependent')


###############################################################################
# Subject-dependent session-independent classification
###############################################################################

if dep_ind:
    preds = []
    Y_test = []
    for subject in subjects:
        for sess in sessions:

            info_str_test = 'DREAMER_' + selected_emotion + f'_subject({subject})_session({sess})_filtered({best_lowcut}, {best_highcut}, {best_order})_samples({best_sample})_start({best_start})_'


            ###############################################################################
            # Data loading
            ###############################################################################

            sess_train = [i for i in range(18) if i != sess[0]]
            info_str_train = 'DREAMER_' + selected_emotion + f'_subject({subject})_session({sess_train})_filtered({best_lowcut}, {best_highcut}, {best_order})_samples({best_sample})_start({best_start})_'
            sess_test = sess
            dataset_train = DREAMERDataset(sets_path+info_str_train, selected_emotion, subjects=subject, sessions=sess_train, samples=best_sample, start=best_start,
                                    lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save)
            dataset_test = DREAMERDataset(sets_path+info_str_test, selected_emotion, subjects=subject, sessions=sess_test, samples=best_sample, start=best_start,
                                    lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save)
            dataset_train_size = len(dataset_train)
            dataset_test_size = len(dataset_test)

            train_indices = list(range(dataset_train_size))
            test_indices = list(range(dataset_test_size))
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

            # Creating data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            train_loader = DataLoader(dataset_train, batch_size=best_batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset_test, batch_size=best_batch_size, sampler=test_sampler)

            print(len(train_indices), 'train samples')
            print(len(test_indices), 'test samples')


            ###############################################################################
            # Model configurations
            ###############################################################################

            model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=best_sample, 
                        dropoutRate=best_dropout, kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2, dropoutType='Dropout').to(device)

            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)


            ###############################################################################
            # Train and test
            ###############################################################################

            for epoch in range(epochs_dep_ind):
                loss = train_f(model, train_loader, optimizer, loss_fn, device)
                acc, loss_test = test_f(model, test_loader, loss_fn, device)
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Train loss: {loss}, Test accuracy: {acc}, Test loss: {loss_test}")

            for batch_index, (X_batch, Y_batch) in enumerate(test_loader):
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                y_pred = model(X_batch)
                _, predicted = torch.max(y_pred.data, 1)
                preds.append(predicted.cpu().numpy())
                _, target = torch.max(Y_batch, 1)
                Y_test.append(target.cpu().numpy())

    classification_accuracy(np.concatenate(preds), np.concatenate(Y_test), names, figs_path, selected_emotion, 'dependent_session_independent')
                            

###############################################################################
# Subject-independent classification
###############################################################################

if independent:
    preds = []
    Y_test = []
    for subject in subjects:

        info_str_test = 'DREAMER_' + selected_emotion + f'_subject({subject})_filtered({best_lowcut}, {best_highcut}, {best_order})_samples({best_sample})_start({best_start})_'


        ###############################################################################
        # Data loading
        ###############################################################################

        subjects_train = [i for i in range(23) if i != subject[0]]
        info_str_train = 'DREAMER_' + selected_emotion + f'_subject({subjects_train})_filtered({best_lowcut}, {best_highcut}, {best_order})_samples({best_sample})_start({best_start})_'
        subjects_test = subject
        dataset_train = DREAMERDataset(sets_path+info_str_train, selected_emotion, subjects=subjects_train, samples=best_sample, start=best_start,
                                lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save)
        dataset_test = DREAMERDataset(sets_path+info_str_test, selected_emotion, subjects=subjects_test, samples=best_sample, start=best_start,
                                lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save)
        dataset_train_size = len(dataset_train)
        dataset_test_size = len(dataset_test)

        train_indices = list(range(dataset_train_size))
        test_indices = list(range(dataset_test_size))
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Creating data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(dataset_train, batch_size=best_batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset_test, batch_size=best_batch_size, sampler=test_sampler)

        print(len(train_indices), 'train samples')
        print(len(test_indices), 'test samples')


        ###############################################################################
        # Model configurations
        ###############################################################################

        model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=best_sample, 
                    dropoutRate=best_dropout, kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2, dropoutType='Dropout').to(device)

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)


        ###############################################################################
        # Train and test
        ###############################################################################

        for epoch in range(epochs_ind):
            loss = train_f(model, train_loader, optimizer, loss_fn, device)
            acc, loss_test = test_f(model, test_loader, loss_fn, device)
            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Train loss: {loss}, Test accuracy: {acc}, Test loss: {loss_test}")

        for batch_index, (X_batch, Y_batch) in enumerate(test_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            y_pred = model(X_batch)
            _, predicted = torch.max(y_pred.data, 1)
            preds.append(predicted.cpu().numpy())
            _, target = torch.max(Y_batch, 1)
            Y_test.append(target.cpu().numpy())

    classification_accuracy(np.concatenate(preds), np.concatenate(Y_test), names, figs_path, selected_emotion, 'independent')

###############################################################################
# Statistical benchmark analysis
###############################################################################

# xDawnRG(dataset, n_components, train_indices, test_indices, chans, samples, names, figs_path, info_str)


