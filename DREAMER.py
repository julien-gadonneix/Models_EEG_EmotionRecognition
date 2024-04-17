###############################################################################
# Imports
###############################################################################

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.EEGModels import EEGNet, EEGNet_SSVEP
from preprocess.preprocess_DREAMER import DREAMERDataset
from tools import train, test, xDawnRG, subject_dependent_classification_accuracy

from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


###############################################################################
# Hyperparameters
###############################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

lowcuts = [.5, .3]
highcuts = [None, 60]
orders = [3, 5]
best_start = 1
# starts = [0, 1, 2, 3, 4]
best_sample = 256
# samples = [128, 256, 512, 1024, 2048]
# subjects = [i for i in range(23)]
subject = None

epochs = 800
batch_size = 128
random_seed= 42
validation_split = .25
test_split = .25
lr = 0.001

best_F1 = 32
best_D = 8
best_F2 = 64
# F1s = [4, 8, 16, 32, 64]
# Ds = [1, 2, 4, 8, 16]
# F2s = [4, 16, 64, 256, 1024]
best_kernLength = 32 # maybe go back to 64 because now f_min = 4Hz
# kernLengths = [16, 32, 64, 128]
best_dropout = .3
# dropouts = [.1, .3, .5]

selected_emotion = 'valence'
class_weights = torch.tensor([1., 1., 1., 1., 1.]).to(device)
names = ['1', '2', '3', '4', '5']
print('Selected emotion:', selected_emotion)

n_components = 2  # pick some components for xDawnRG

figs_path = './figs/'
sets_path = './sets/'
save_figs = True
np.random.seed(random_seed)


###############################################################################
# Search for optimal hyperparameters
###############################################################################

# preds_total = []
# Y_test_total = []
for order, highcut, lowcut in zip(orders, highcuts, lowcuts):
      info_str = 'DREAMER_' + selected_emotion + f'_subject({subject})_filtered({lowcut}, {highcut}, {order})_samples({best_sample})_start({best_start})_'


      ###############################################################################
      # Data loading
      ###############################################################################

      dataset = DREAMERDataset(sets_path+info_str, selected_emotion, subject=subject, samples=best_sample, start=best_start, lowcut=lowcut, highcut=highcut, order=order)
      dataset_size = len(dataset)

      indices = list(range(dataset_size))
      np.random.shuffle(indices)
      split_val = int(np.floor(validation_split * dataset_size))
      split_test = int(np.floor((test_split+validation_split) * dataset_size))
      train_indices, test_indices, val_indices = indices[split_test:], indices[split_val:split_test], indices[:split_val]

      # Creating data samplers and loaders:
      train_sampler = SubsetRandomSampler(train_indices)
      valid_sampler = SubsetRandomSampler(val_indices)
      test_sampler = SubsetRandomSampler(test_indices)
      train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
      validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
      test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

      print(len(train_indices), 'train samples')
      print(len(test_indices), 'test samples')


      ###############################################################################
      # Model configurations
      ###############################################################################

      chans = dataset.data[0].shape[1]
      nb_classes = dataset.targets[0].shape[0]
      best_model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=best_sample, 
                  dropoutRate=best_dropout, kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2, dropoutType='Dropout').to(device)

      # set a valid path for your system to record model checkpoints
      checkpointer = './tmp/checkpoint_' + info_str + best_model.name + '.pth'

      # compile the model and set the optimizers
      loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
      optimizer = torch.optim.Adam(best_model.parameters(), lr=lr)


      ###############################################################################
      # Train and test
      ###############################################################################

      train(best_model, epochs, train_loader, validation_loader, optimizer, loss_fn, checkpointer,
            device, figs_path, info_str, save_figs)

      # load optimal weights
      best_model.load_state_dict(torch.load(checkpointer))

      preds, Y_test = test(best_model, test_loader, names, figs_path, device, info_str, save_figs)
# test(model2, test_loader, names, figs_path, device, info_str)

# preds_total.append(preds)
# Y_test_total.append(Y_test)

# preds_total = np.concatenate(preds_total)
# Y_test_total = np.concatenate(Y_test_total)
# subject_dependent_classification_accuracy(preds_total, Y_test_total, names, figs_path, selected_emotion)

###############################################################################
# Statistical benchmark analysis
###############################################################################

# xDawnRG(dataset, n_components, train_indices, test_indices, chans, samples, names, figs_path, info_str)


