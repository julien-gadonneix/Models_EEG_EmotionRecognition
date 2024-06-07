###############################################################################
# Imports
###############################################################################

import numpy as np
import torch
from pathlib import Path

from models.EEGModels import EEGNet, EEGNet_SSVEP, CapsEEGNet, TCNet
from preprocess.preprocess_DREAMER import DREAMERDataset
from tools import train_f, test_f, xDawnRG, classification_accuracy, draw_loss

from torch.utils.data import DataLoader, SubsetRandomSampler


###############################################################################
# Hyperparameters
###############################################################################

selected_emotion = 'valence'
selected_model = 'TCNet'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)
is_ok = device.type != 'mps'

best_highcut = None
best_lowcut = .5
best_order = 3
best_type = 'butter'
best_start = 1
best_sample = 128
subjects = [[i] for i in range(23)]
sessions = [[i] for i in range(18)]
best_tfrs = {'EEGNet': None, 'CapsEEGNet': None, 'TCNet': {'freqs': np.arange(2, 50), 'output': 'power'}}
best_tfr = best_tfrs[selected_model]

epochs_dep_mixs = {'EEGNet': 800, 'CapsEEGNet': 300, 'TCNet': 30}
epochs_dep_mix = epochs_dep_mixs[selected_model]
epochs_dep_ind = 800
epochs_ind = 20
test_split = .25

best_lrs = {'EEGNet': 0.001, 'CapsEEGNet': 0.01, 'TCNet': 0.000001}
best_lr = best_lrs[selected_model]
best_batch_sizes = {'EEGNet': 128, 'CapsEEGNet': 16, 'TCNet': 128}
best_batch_size = best_batch_sizes[selected_model]
best_F1 = 64
best_D = 8
best_F2 = 64
best_kernLengths = {'arousal': 20, 'dominance': 12, 'valence': 12} # maybe go back to 64 for f_min = 2Hz
best_kernLength = best_kernLengths[selected_emotion]
best_dropout = .1
best_norm_rate = .25
best_nr = 1.


best_group_classes = False
best_adapt_classWeights = False
if best_group_classes:
      class_weights = torch.tensor([1., 1.]).to(device)
      names = ['Low', 'High']
else:
      class_weights = torch.tensor([1., 1., 1., 1., 1.]).to(device)
      names = ['1', '2', '3', '4', '5']

n_components = 2  # pick some components for xDawnRG
nb_classes = len(names)
chans = 14

cur_dir = Path(__file__).resolve().parent
figs_path = str(cur_dir) + '/figs/'
sets_path = str(cur_dir) + '/sets/'
models_path = str(cur_dir) + '/tmp/'
save = False

dep_mix = True
dep_ind = False
independent = False


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
                                 lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save, group_classes=best_group_classes, tfr=best_tfr)
        dataset_size = len(dataset)

        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split_test = int(np.floor(test_split * dataset_size))
        train_indices, test_indices = indices[split_test:], indices[:split_test]

        # Creating data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(dataset, batch_size=best_batch_size, sampler=train_sampler, pin_memory=True)
        test_loader = DataLoader(dataset, batch_size=best_batch_size, sampler=test_sampler, pin_memory=True)
        print(len(train_indices), 'train samples')
        print(len(test_indices), 'test samples')


        ###############################################################################
        # Model configurations
        ###############################################################################

        if selected_model == 'CapsEEGNet':
            model = CapsEEGNet(nb_classes=nb_classes).to(device=device)
        elif selected_model == 'EEGNet':
            model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=best_sample, dropoutRate=best_dropout,
                           kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2,
                           norm_rate=best_norm_rate, nr=best_nr, dropoutType='Dropout').to(device=device, memory_format=torch.channels_last)
        elif selected_model == 'TCNet':
            model = TCNet(nb_classes=nb_classes, device=device, Chans=chans).to(device=device)
        else:
            raise ValueError('Invalid model selected')

        loss_fn = torch.nn.CrossEntropyLoss(weight=dataset.class_weights).to(device) if best_adapt_classWeights else torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=is_ok)

        torch.backends.cudnn.benchmark = True


        ###############################################################################
        # Train and test
        ###############################################################################

        losses_train = []
        losses_test = []
        for epoch in range(epochs_dep_mix):
            loss = train_f(model, train_loader, optimizer, loss_fn, scaler, device, is_ok, selected_model)
            losses_train.append(loss)
            acc, loss_test = test_f(model, test_loader, loss_fn, device, is_ok, selected_model)
            losses_test.append(loss_test)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train loss: {loss}, Test accuracy: {acc}, Test loss: {loss_test}")
        draw_loss(losses_train, losses_test, figs_path, selected_emotion, str(subject))

        with torch.no_grad():
            for batch_index, (X_batch, Y_batch) in enumerate(test_loader):
                if selected_model not in ['CapsEEGNet', 'TCNet']:
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
                                    lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save, group_classes=best_group_classes, tfr=best_tfr)
            dataset_test = DREAMERDataset(sets_path+info_str_test, selected_emotion, subjects=subject, sessions=sess_test, samples=best_sample, start=best_start,
                                    lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save, group_classes=best_group_classes, tfr=best_tfr)
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
            test_loader = DataLoader(dataset_test, batch_size=best_batch_size, sampler=test_sampler, pin_memory=True)

            print(len(train_indices), 'train samples')
            print(len(test_indices), 'test samples')


            ###############################################################################
            # Model configurations
            ###############################################################################

            if selected_model == 'CapsEEGNet':
                model = CapsEEGNet(nb_classes=nb_classes).to(device=device)
            elif selected_model == 'EEGNet':
                model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=best_sample, dropoutRate=best_dropout,
                               kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2,
                               norm_rate=best_norm_rate, nr=best_nr, dropoutType='Dropout').to(device=device, memory_format=torch.channels_last)
            elif selected_model == 'TCNet':
                model = TCNet(nb_classes=nb_classes).to(device=device)
            else:
                raise ValueError('Invalid model selected')

            loss_fn = torch.nn.CrossEntropyLoss(weight=dataset.class_weights).to(device) if best_adapt_classWeights else torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            scaler = torch.cuda.amp.GradScaler(enabled=is_ok)

            torch.backends.cudnn.benchmark = True


            ###############################################################################
            # Train and test
            ###############################################################################

            for epoch in range(epochs_dep_ind):
                loss = train_f(model, train_loader, optimizer, loss_fn, scaler, device, is_ok, selected_model)
                acc, loss_test = test_f(model, test_loader, loss_fn, device, is_ok, selected_model)
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Train loss: {loss}, Test accuracy: {acc}, Test loss: {loss_test}")

            with torch.no_grad():
                for batch_index, (X_batch, Y_batch) in enumerate(test_loader):
                    if selected_model not in ['CapsEEGNet', 'TCNet']:
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
                                lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save, group_classes=best_group_classes, tfr=best_tfr)
        dataset_test = DREAMERDataset(sets_path+info_str_test, selected_emotion, subjects=subjects_test, samples=best_sample, start=best_start,
                                lowcut=best_lowcut, highcut=best_highcut, order=best_order, type=best_type, save=save, group_classes=best_group_classes, tfr=best_tfr)
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
        test_loader = DataLoader(dataset_test, batch_size=best_batch_size, sampler=test_sampler, pin_memory=True)

        print(len(train_indices), 'train samples')
        print(len(test_indices), 'test samples')


        ###############################################################################
        # Model configurations
        ###############################################################################

        if selected_model == 'CapsEEGNet':
            model = CapsEEGNet(nb_classes=nb_classes).to(device=device)
        elif selected_model == 'EEGNet':
            model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=best_sample, dropoutRate=best_dropout,
                           kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2,
                           norm_rate=best_norm_rate, nr=best_nr, dropoutType='Dropout').to(device=device, memory_format=torch.channels_last)
        elif selected_model == 'TCNet':
            model = TCNet(nb_classes=nb_classes).to(device=device)
        else:
            raise ValueError('Invalid model selected')

        loss_fn = torch.nn.CrossEntropyLoss(weight=dataset.class_weights).to(device) if best_adapt_classWeights else torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=is_ok)

        torch.backends.cudnn.benchmark = True


        ###############################################################################
        # Train and test
        ###############################################################################

        for epoch in range(epochs_ind):
            loss = train_f(model, train_loader, optimizer, loss_fn, scaler, device, is_ok, selected_model)
            acc, loss_test = test_f(model, test_loader, loss_fn, device, is_ok, selected_model)
            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Train loss: {loss}, Test accuracy: {acc}, Test loss: {loss_test}")

        with torch.no_grad():
            for batch_index, (X_batch, Y_batch) in enumerate(test_loader):
                if selected_model not in ['CapsEEGNet', 'TCNet']:
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

    classification_accuracy(np.concatenate(preds), np.concatenate(Y_test), names, figs_path, selected_emotion, 'independent')

###############################################################################
# Statistical benchmark analysis
###############################################################################

# xDawnRG(dataset, n_components, train_indices, test_indices, chans, samples, names, figs_path, info_str)


