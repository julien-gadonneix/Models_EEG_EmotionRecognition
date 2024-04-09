###############################################################################
# Imports
###############################################################################

import numpy as np
import torch
from matplotlib import pyplot as plt

from models.EEGModels import EEGNet
from preprocess.preprocess_DREAMER import DREAMERDataset
from tools import train, test, xDawnRG

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SubsetRandomSampler


###############################################################################
# Hyperparameters
###############################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

epochs = 1000
batch_size = 64
random_seed= 42
validation_split = .25
test_split = .25
lr = 0.001

F1 = 8
D = 2
F2 = 16
kernLength = 32
dropout = 0.5
samples = 128

selected_emotion = 'valence'
class_weights = torch.tensor([1., 1., 1., 1., 1.]).to(device) # to be adjusted
names = ['1', '2', '3', '4', '5']
print('Selected emotion:', selected_emotion)

# set a valid path for your system to record model checkpoints
checkpointer = './tmp/checkpoint_DREAMER_' + selected_emotion + '.pth'
figs_path = './figs/'

n_components = 2  # pick some components for xDwanRG


###############################################################################
# Data loading
###############################################################################

dataset = DREAMERDataset(selected_emotion, samples=samples)
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

chans = len(dataset.eeg_electrodes)
samples = dataset.samples
nb_classes = dataset.targets[0].shape[0]
model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples, 
               dropoutRate=dropout, kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType='Dropout').to(device)

# compile the model and set the optimizers
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


###############################################################################
# Train and test
###############################################################################

train(model, epochs, train_loader, validation_loader, optimizer, loss_fn, checkpointer, device)

# load optimal weights
model.load_state_dict(torch.load(checkpointer))

test(model, test_loader, selected_emotion, names, figs_path, device)
xDawnRG(dataset, n_components, train_indices, test_indices, chans, samples, selected_emotion, names, figs_path)


