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

lowcut = 0.3
highcut = 60.
order = 5
start = 0
samples = 256
subject = None

epochs = 500
batch_size = 128
random_seed= 42
validation_split = .25
test_split = .25
lr = 0.001

F1 = 8
D = 2
F2 = 16
kernLength = 64
dropout = 0.3

selected_emotion = 'valence'
class_weights = torch.tensor([1., 1., 1., 1., 1.]).to(device) # to be adjusted
names = ['1', '2', '3', '4', '5']
print('Selected emotion:', selected_emotion)

n_components = 2  # pick some components for xDwanRG

figs_path = './figs/'
sets_path = './sets/'
info_str = 'DREAMER_' + selected_emotion + f'_subject({subject})_filtered({lowcut}, {highcut}, {order})_samples({samples})_start({start})_'


###############################################################################
# Data loading
###############################################################################

dataset = DREAMERDataset(sets_path+info_str, selected_emotion, subject=subject, samples=samples, start=start, lowcut=lowcut, highcut=highcut, order=order)
dataset_size = len(dataset)

indices = list(range(dataset_size))
np.random.seed(random_seed)
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
model1 = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples, 
               dropoutRate=dropout, kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType='Dropout').to(device)
model2 = EEGNet_SSVEP(nb_classes=nb_classes, Chans=chans, Samples=samples, 
               dropoutRate=dropout, kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType='Dropout').to(device)

# set a valid path for your system to record model checkpoints
checkpointer1 = './tmp/checkpoint_' + info_str + model1.name + '.pth'
checkpointer2 = './tmp/checkpoint_' + info_str + model2.name + '.pth'

# compile the model and set the optimizers
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model1.parameters(), lr=lr)


###############################################################################
# Train and test
###############################################################################

train(model1, epochs, train_loader, validation_loader, optimizer1, loss_fn, checkpointer1,
      device, figs_path, info_str)
train(model2, epochs, train_loader, validation_loader, optimizer2, loss_fn, checkpointer2,
      device, figs_path, info_str)

# load optimal weights
model1.load_state_dict(torch.load(checkpointer1))
model2.load_state_dict(torch.load(checkpointer2))

test(model1, test_loader, names, figs_path, device, info_str)
test(model2, test_loader, names, figs_path, device, info_str)
xDawnRG(dataset, n_components, train_indices, test_indices, chans, samples, names, figs_path, info_str)


