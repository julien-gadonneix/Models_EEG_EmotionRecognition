###############################################################################
# Imports
###############################################################################

import numpy as np
import torch

from models.EEGModels import EEGNet, EEGNet_SSVEP
from preprocess.preprocess_DREAMER import DREAMERDataset
from tools import train, test, xDawnRG

from torch.utils.data import DataLoader, SubsetRandomSampler


###############################################################################
# Hyperparameters
###############################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

lowcut = .3
highcut = 60.
order = 5
start = 0
best_sample = 256
# samples = [128, 256, 512, 1024, 2048]
subject = None

epochs = 1500
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
dropouts = [.1, .3, .5, .7]

selected_emotion = 'valence'
class_weights = torch.tensor([1., 1., 1., 1., 1.]).to(device) # to be adjusted
names = ['1', '2', '3', '4', '5']
print('Selected emotion:', selected_emotion)

n_components = 2  # pick some components for xDawnRG

figs_path = './figs/'
sets_path = './sets/'
np.random.seed(random_seed)


###############################################################################
# Search for optimal hyperparameters
###############################################################################

for dropout in dropouts:
      info_str = 'DREAMER_' + selected_emotion + f'_subject({subject})_filtered({lowcut}, {highcut}, {order})_samples({best_sample})_start({start})_'


      ###############################################################################
      # Data loading
      ###############################################################################

      dataset = DREAMERDataset(sets_path+info_str, selected_emotion, subject=subject, samples=best_sample, start=start, lowcut=lowcut, highcut=highcut, order=order)
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
                  dropoutRate=dropout, kernLength=best_kernLength, F1=best_F1, D=best_D, F2=best_F2, dropoutType='Dropout').to(device)
      # model2 = EEGNet_SSVEP(nb_classes=nb_classes, Chans=chans, Samples=samples, 
      #                dropoutRate=dropout, kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType='Dropout').to(device)

      # set a valid path for your system to record model checkpoints
      checkpointer1 = './tmp/checkpoint_' + info_str + best_model.name + '.pth'
      # checkpointer2 = './tmp/checkpoint_' + info_str + model2.name + '.pth'

      # compile the model and set the optimizers
      loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
      optimizer1 = torch.optim.Adam(best_model.parameters(), lr=lr)
      # optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)


      ###############################################################################
      # Train and test
      ###############################################################################

      train(best_model, epochs, train_loader, validation_loader, optimizer1, loss_fn, checkpointer1,
            device, figs_path, info_str)
      # train(model2, epochs, train_loader, validation_loader, optimizer2, loss_fn, checkpointer2,
      #       device, figs_path, info_str)

      # load optimal weights
      best_model.load_state_dict(torch.load(checkpointer1))
      # model2.load_state_dict(torch.load(checkpointer2))

      test(best_model, test_loader, names, figs_path, device, info_str)
      # test(model2, test_loader, names, figs_path, device, info_str)


###############################################################################
# Statistical benchmark analysis
###############################################################################

# xDawnRG(dataset, n_components, train_indices, test_indices, chans, samples, names, figs_path, info_str)


