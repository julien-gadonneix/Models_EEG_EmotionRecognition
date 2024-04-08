"""
 Sample script using EEGNet to classify Event-Related Potential (ERP) EEG data
 from a four-class classification task, using the sample dataset provided in
 the MNE [1, 2] package:
     https://martinos.org/mne/stable/manual/sample_dataset.html#ch-sample-data
   
 The four classes used from this dataset are:
     LA: Left-ear auditory stimulation
     RA: Right-ear auditory stimulation
     LV: Left visual field stimulation
     RV: Right visual field stimulation

 The code to process, filter and epoch the data are originally from Alexandre
 Barachant's PyRiemann [3] package, released under the BSD 3-clause. A copy of 
 the BSD 3-clause license has been provided together with this software to 
 comply with software licensing requirements. 
 
 When you first run this script, MNE will download the dataset and prompt you
 to confirm the download location (defaults to ~/mne_data). Follow the prompts
 to continue. The dataset size is approx. 1.5GB download. 
 
 For comparative purposes you can also compare EEGNet performance to using 
 Riemannian geometric approaches with xDAWN spatial filtering [4-8] using 
 PyRiemann (code provided below).

 [1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck,
     L. Parkkonen, M. Hämäläinen, MNE software for processing MEG and EEG data, 
     NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119.

 [2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, 
     R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data 
     analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013.

 [3] https://github.com/alexandrebarachant/pyRiemann. 

 [4] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information Geometry"
     arXiv:1409.0107. link

 [5] M. Congedo, A. Barachant, A. Andreev ,"A New generation of Brain-Computer 
     Interface Based on Riemannian Geometry", arXiv: 1310.8115.

 [6] A. Barachant and S. Bonnet, "Channel selection procedure using riemannian 
     distance for BCI applications," in 2011 5th International IEEE/EMBS 
     Conference on Neural Engineering (NER), 2011, 348-351.

 [7] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Multiclass 
     Brain-Computer Interface Classification by Riemannian Geometry,” in IEEE 
     Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

 [8] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Classification of 
     covariance matrices using a Riemannian-based kernel for BCI applications“, 
     in NeuroComputing, vol. 112, p. 172-178, 2013.


 Portions of this project are works of the United States Government and are not
 subject to domestic copyright protection under 17 USC Sec. 105.  Those 
 portions are released world-wide under the terms of the Creative Commons Zero 
 1.0 (CC0) license.  
 
 Other portions of this project are subject to domestic copyright protection 
 under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0 
 license.  The complete text of the license governing this material is in 
 the file labeled LICENSE.TXT that is a part of this project's official 
 distribution. 
"""

import numpy as np

# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from models.EEGModels import EEGNet
# from tensorflow.keras import utils as np_utils
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras import backend as K
import torch

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
# from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
# K.set_image_data_format('channels_last')

##################### Process, filter and epoch the data ######################
# data_path = sample.data_path()
data_path = '/Users/juliengadonneix/Desktop/stage 3a/data/mne_data'

# Set parameters and read data
data_path = str(data_path)
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
# X = epochs.get_data()*1000 # format is in (trials, channels, samples)
X = epochs.get_data(copy=False) * 1000
y = labels

kernels, chans, samples = 1, 60, 151

# take 50/25/25 percent of the data to train/validate/test
X_train      = X[0:144,]
Y_train      = y[0:144]
X_validate   = X[144:216,]
Y_validate   = y[144:216]
X_test       = X[216:,]
Y_test       = y[216:]

############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
# Y_train      = np_utils.to_categorical(Y_train-1)
# Y_validate   = np_utils.to_categorical(Y_validate-1)
# Y_test       = np_utils.to_categorical(Y_test-1)
Y_train = torch.nn.functional.one_hot(torch.tensor(Y_train-1)).float()
Y_validate = torch.nn.functional.one_hot(torch.tensor(Y_validate-1)).float()
Y_test = torch.nn.functional.one_hot(torch.tensor(Y_test-1)).float()

# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
# X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
# X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
# X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
X_train = torch.tensor(X_train.reshape(X_train.shape[0], kernels, chans, samples), dtype=torch.float32)
X_validate = torch.tensor(X_validate.reshape(X_validate.shape[0], kernels, chans, samples), dtype=torch.float32)
X_test = torch.tensor(X_test.reshape(X_test.shape[0], kernels, chans, samples), dtype=torch.float32)
   
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes=4, Chans=chans, Samples=samples, 
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16, dropoutType='Dropout')

# compile the model and set the optimizers
# model.compile(loss='categorical_crossentropy', optimizer='adam', 
#               metrics = ['accuracy'])
class_weights = torch.tensor([1., 1., 1., 1.])
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# count number of parameters in the model
# numParams = model.count_params()  
numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)

# set a valid path for your system to record model checkpoints
# checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
#                                save_best_only=True)
checkpointer = './tmp/checkpoint.pth'

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during 
# optimization to balance it out. This data is approximately balanced so we 
# don't need to do this, but is shown here for illustration/completeness. 
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
# class_weights = {0:1, 1:1, 2:1, 3:1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
# Riemannian geometry classification (below)
################################################################################
# fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
#                         verbose = 2, validation_data=(X_validate, Y_validate),
#                         callbacks=[checkpointer], class_weight = class_weights)
min_val_loss = float('inf')
for epoch in range(300):
    avg_loss = 0.
    model.train()
    for i in range(0, X_train.shape[0], 16):
        optimizer.zero_grad()
        bs = min(16, X_train.shape[0] - i)
        X_batch = X_train[i:i+bs]
        Y_batch = Y_train[i:i+bs]
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, Y_batch)
        avg_loss += loss.item() * bs
        loss.backward()
        optimizer.step()
    avg_loss /= X_train.shape[0]

    model.eval()
    with torch.no_grad():
        y_pred = model(X_validate)
        val_loss = loss_fn(y_pred, Y_validate).item()
        val_acc = (y_pred.argmax(dim=-1) == Y_validate.argmax(dim=-1)).float().mean().item()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), checkpointer)

    print(f'Epoch {epoch}: Train loss: {avg_loss} Validation loss: {val_loss} Validation accuracy: {val_acc}')

# load optimal weights
# model.load_weights('/tmp/checkpoint.h5')
model.load_state_dict(torch.load(checkpointer))

###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

# probs       = model.predict(X_test)
probs       = model(X_test).detach().numpy()
preds       = probs.argmax(axis = -1)  
# acc         = np.mean(preds == Y_test.argmax(axis=-1))
acc = np.mean(preds == Y_test.argmax(axis=-1).numpy())
print("Classification accuracy: %f " % (acc))


############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in 
# the tangent space with a logistic regression

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg     = np.zeros(len(Y_test))

# reshape back to (trials, channels, samples)
X_train      = X_train.reshape(X_train.shape[0], chans, samples)
X_test       = X_test.reshape(X_test.shape[0], chans, samples)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
# clf.fit(X_train, Y_train.argmax(axis = -1))
# preds_rg     = clf.predict(X_test)
clf.fit(X_train.numpy(), Y_train.argmax(axis=-1).numpy())
preds_rg = clf.predict(X_test.numpy())

# Printing the results
# acc2         = np.mean(preds_rg == Y_test.argmax(axis = -1))
acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1).numpy())
print("Classification accuracy: %f " % (acc2))

# plot the confusion matrices for both classifiers
names        = ['audio left', 'audio right', 'vis left', 'vis right']
plt.figure(0)
# plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
ConfusionMatrixDisplay(confusion_matrix(preds, Y_test.argmax(axis = -1).numpy()), display_labels=names).plot()
plt.title('EEGNet-8,2')

plt.figure(1)
# plot_confusion_matrix(preds_rg, Y_test.argmax(axis = -1), names, title = 'xDAWN + RG')
ConfusionMatrixDisplay(confusion_matrix(preds_rg, Y_test.argmax(axis = -1).numpy()), display_labels=names).plot()
plt.title('xDAWN + RG')

plt.show()



