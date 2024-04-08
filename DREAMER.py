import numpy as np
import torch
import scipy.io

from models.EEGModels import EEGNet

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

data_path = '/Users/juliengadonneix/Desktop/stage 3a/data/DREAMER/'
mat = scipy.io.loadmat(data_path + 'DREAMER.mat')

data, eeg_sr, ecg_sr, eeg_electrodes, n_subjects, n_videos, _, _, _, _  = mat['DREAMER'][0, 0]
n_subjects = n_subjects[0, 0]
n_videos = n_videos[0, 0]

samples = np.inf
for i in range(n_subjects):
    age, gender, eeg, ecg, val, aro, dom = data[0, i][0][0]
    baseline_eeg, stimuli_eeg = eeg[0, 0]
    for j in range(n_videos):
        stimuli_eeg_j = stimuli_eeg[j, 0]
        samples = min(stimuli_eeg_j.shape[0], samples)

X = []
y_val = []
y_aro = []
y_dom = []
for i in range(n_subjects):
    age, gender, eeg, ecg, val, aro, dom = data[0, i][0][0]
    baseline_eeg, stimuli_eeg = eeg[0, 0]
    for j in range(n_videos):
        stimuli_eeg_j = stimuli_eeg[j, 0]
        X.append(torch.tensor(stimuli_eeg_j[-samples:, :].T * 1000, dtype=torch.float32)) # scale by 1000 due to scaling sensitivity in DL
        y_val.append(val[j, 0]-1)
        y_aro.append(aro[j, 0]-1)
        y_dom.append(dom[j, 0]-1)
X = torch.stack(X)
y_val = torch.tensor(y_val)
y_aro = torch.tensor(y_aro)
y_dom = torch.tensor(y_dom)

kernels, chans = 1, 14
p20 = n_subjects * n_videos // 5

perm_indices = torch.randperm(X.shape[0])
X = X[perm_indices]
y_val = y_val[perm_indices]
y_aro = y_aro[perm_indices]
y_dom = y_dom[perm_indices]

# take 50/25/25 percent of the data to train/validate/test
X_train      = X[0:4*p20,:,:]
Y_train_val      = y_val[0:4*p20]
Y_train_aro      = y_aro[0:4*p20]
Y_train_dom      = y_dom[0:4*p20]
X_validate   = X[4*p20:int(4.5*p20),:,:]
Y_validate_val   = y_val[4*p20:int(4.5*p20)]
Y_validate_aro   = y_aro[4*p20:int(4.5*p20)]
Y_validate_dom   = y_dom[4*p20:int(4.5*p20)]
X_test       = X[int(4.5*p20):,:,:]
Y_test_val       = y_val[int(4.5*p20):]
Y_test_aro       = y_aro[int(4.5*p20):]
Y_test_dom       = y_dom[int(4.5*p20):]


############################# EEGNet portion ##################################

Y_train_val = torch.nn.functional.one_hot(Y_train_val).float().to(device)
Y_validate_val = torch.nn.functional.one_hot(Y_validate_val).float().to(device)
Y_test_val = torch.nn.functional.one_hot(Y_test_val).float().to(device)
Y_train_aro = torch.nn.functional.one_hot(Y_train_aro).float().to(device)
Y_validate_aro = torch.nn.functional.one_hot(Y_validate_aro).float().to(device)
Y_test_aro = torch.nn.functional.one_hot(Y_test_aro).float().to(device)
Y_train_dom = torch.nn.functional.one_hot(Y_train_dom).float().to(device)
Y_validate_dom = torch.nn.functional.one_hot(Y_validate_dom).float().to(device)
Y_test_dom = torch.nn.functional.one_hot(Y_test_dom).float().to(device)

X_train = X_train.reshape(X_train.shape[0], kernels, chans, samples).float().to(device)
X_validate = X_validate.reshape(X_validate.shape[0], kernels, chans, samples).float().to(device)
X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples).float().to(device)
   
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
model_val = EEGNet(nb_classes=5, Chans=chans, Samples=samples, 
               dropoutRate=0.5, kernLength=32, F1=32, D=8, F2=256, dropoutType='Dropout').to(device)
model_aro = EEGNet(nb_classes=5, Chans=chans, Samples=samples, 
               dropoutRate=0.5, kernLength=32, F1=32, D=8, F2=256, dropoutType='Dropout').to(device)
model_dom = EEGNet(nb_classes=5, Chans=chans, Samples=samples, 
               dropoutRate=0.5, kernLength=32, F1=32, D=8, F2=256, dropoutType='Dropout').to(device)

# compile the model and set the optimizers
class_weights = torch.tensor([1., 1., 1., 1., 1.]).to(device) # to be adjusted
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
lr = 0.00001
optimizer_val = torch.optim.Adam(model_val.parameters(), lr=lr)
optimizer_aro = torch.optim.Adam(model_aro.parameters(), lr=lr)
optimizer_dom = torch.optim.Adam(model_dom.parameters(), lr=lr)

# set a valid path for your system to record model checkpoints
checkpointer_val = './tmp/checkpoint_DREAMER_val.pth'
checkpointer_aro = './tmp/checkpoint_DREAMER_aro.pth'
checkpointer_dom = './tmp/checkpoint_DREAMER_dom.pth'


###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during 
# optimization to balance it out. This data is approximately balanced so we 
# don't need to do this, but is shown here for illustration/completeness. 
###############################################################################

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
# Riemannian geometry classification (below)
################################################################################
min_val_loss_val = float('inf')
min_val_loss_aro = float('inf')
min_val_loss_dom = float('inf')
for epoch in range(300):
    avg_loss_val = 0.
    avg_loss_aro = 0.
    avg_loss_dom = 0.
    model_val.train()
    model_aro.train()
    model_dom.train()
    for i in range(0, X_train.shape[0], 32):
        optimizer_val.zero_grad()
        optimizer_aro.zero_grad()
        optimizer_dom.zero_grad()
        bs = min(32, X_train.shape[0] - i)
        X_batch = X_train[i:i+bs]
        Y_batch_val = Y_train_val[i:i+bs]
        Y_batch_aro = Y_train_aro[i:i+bs]
        Y_batch_dom = Y_train_dom[i:i+bs]
        y_pred_val = model_val(X_batch)
        y_pred_aro = model_aro(X_batch)
        y_pred_dom = model_dom(X_batch)
        loss_val = loss_fn(y_pred_val, Y_batch_val)
        loss_aro = loss_fn(y_pred_aro, Y_batch_aro)
        loss_dom = loss_fn(y_pred_dom, Y_batch_dom)
        avg_loss_val += loss_val.item() * bs
        avg_loss_aro += loss_aro.item() * bs
        avg_loss_dom += loss_dom.item() * bs
        loss_val.backward()
        loss_aro.backward()
        loss_dom.backward()
        optimizer_val.step()
        optimizer_aro.step()
        optimizer_dom.step()
    avg_loss_val /= X_train.shape[0]
    avg_loss_aro /= X_train.shape[0]
    avg_loss_dom /= X_train.shape[0]

    model_val.eval()
    model_aro.eval()
    model_dom.eval()
    with torch.no_grad():
        y_pred_val = model_val(X_validate)
        y_pred_aro = model_aro(X_validate)
        y_pred_dom = model_dom(X_validate)
        val_loss_val = loss_fn(y_pred_val, Y_validate_val).item()
        val_loss_aro = loss_fn(y_pred_aro, Y_validate_aro).item()
        val_loss_dom = loss_fn(y_pred_dom, Y_validate_dom).item()
        val_acc_val = (y_pred_val.argmax(dim=-1) == Y_validate_val.argmax(dim=-1)).float().mean().item()
        val_acc_aro = (y_pred_aro.argmax(dim=-1) == Y_validate_aro.argmax(dim=-1)).float().mean().item()
        val_acc_dom = (y_pred_dom.argmax(dim=-1) == Y_validate_dom.argmax(dim=-1)).float().mean().item()
        if val_loss_val < min_val_loss_val:
            min_val_loss_val = val_loss_val
            torch.save(model_val.state_dict(), checkpointer_val)
        if val_loss_aro < min_val_loss_aro:
            min_val_loss_aro = val_loss_aro
            torch.save(model_aro.state_dict(), checkpointer_aro)
        if val_loss_dom < min_val_loss_dom:
            min_val_loss_dom = val_loss_dom
            torch.save(model_dom.state_dict(), checkpointer_dom)

    print(f'Epoch {epoch} (Valence): Train loss: {avg_loss_val} Validation loss: {val_loss_val} Validation accuracy: {val_acc_val}')
    print(f'Epoch {epoch} (Arousal): Train loss: {avg_loss_aro} Validation loss: {val_loss_aro} Validation accuracy: {val_acc_aro}')
    print(f'Epoch {epoch} (Dominance): Train loss: {avg_loss_dom} Validation loss: {val_loss_dom} Validation accuracy: {val_acc_dom}')

# load optimal weights
model_val.load_state_dict(torch.load(checkpointer_val))
model_aro.load_state_dict(torch.load(checkpointer_aro))
model_dom.load_state_dict(torch.load(checkpointer_dom))

###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

###############################################################################
# make prediction on test set.
###############################################################################

probs_val       = model_val(X_test).detach().cpu().numpy()
probs_aro       = model_aro(X_test).detach().cpu().numpy()
probs_dom       = model_dom(X_test).detach().cpu().numpy()
preds_val       = probs_val.argmax(axis = -1)
preds_aro       = probs_aro.argmax(axis = -1)
preds_dom       = probs_dom.argmax(axis = -1)
acc_val = np.mean(preds_val == Y_test_val.argmax(axis=-1).cpu().numpy())
acc_aro = np.mean(preds_aro == Y_test_aro.argmax(axis=-1).cpu().numpy())
acc_dom = np.mean(preds_dom == Y_test_dom.argmax(axis=-1).cpu().numpy())
print("Classification accuracy on valence: %f " % (acc_val))
print("Classification accuracy on arousal: %f " % (acc_aro))
print("Classification accuracy on dominance: %f " % (acc_dom))


############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in 
# the tangent space with a logistic regression

n_components = 2  # pick some components

# set up sklearn pipeline
clf_val = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())
clf_aro = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())
clf_dom = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg_val     = np.zeros(len(Y_test_val))
preds_rg_aro     = np.zeros(len(Y_test_aro))
preds_rg_dom     = np.zeros(len(Y_test_dom))

# reshape back to (trials, channels, samples)
X_train      = X_train.reshape(X_train.shape[0], chans, samples)
X_test       = X_test.reshape(X_test.shape[0], chans, samples)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf_val.fit(X_train.numpy(), Y_train_val.argmax(axis=-1).cpu().numpy())
clf_aro.fit(X_train.numpy(), Y_train_aro.argmax(axis=-1).cpu().numpy())
clf_dom.fit(X_train.numpy(), Y_train_dom.argmax(axis=-1).cpu().numpy())
preds_rg_val = clf_val.predict(X_test.numpy())
preds_rg_aro = clf_aro.predict(X_test.numpy())
preds_rg_dom = clf_dom.predict(X_test.numpy())

# Printing the results
acc2_val = np.mean(preds_rg_val == Y_test_val.argmax(axis=-1).cpu().numpy())
acc2_aro = np.mean(preds_rg_aro == Y_test_aro.argmax(axis=-1).cpu().numpy())
acc2_dom = np.mean(preds_rg_dom == Y_test_dom.argmax(axis=-1).cpu().numpy())
print("Classification accuracy: %f " % (acc2_val))
print("Classification accuracy: %f " % (acc2_aro))
print("Classification accuracy: %f " % (acc2_dom))

# plot the confusion matrices for both classifiers
names = ['1', '2', '3', '4', '5']
ConfusionMatrixDisplay(confusion_matrix(preds_val, Y_test_val.argmax(axis = -1).cpu().numpy()), display_labels=names).plot()
plt.title('EEGNet-8,2 valence')
plt.show()
ConfusionMatrixDisplay(confusion_matrix(preds_aro, Y_test_aro.argmax(axis = -1).cpu().numpy()), display_labels=names).plot()
plt.title('EEGNet-8,2 arousal')
plt.show()
ConfusionMatrixDisplay(confusion_matrix(preds_dom, Y_test_dom.argmax(axis = -1).cpu().numpy()), display_labels=names).plot()
plt.title('EEGNet-8,2 dominance')
plt.show()

ConfusionMatrixDisplay(confusion_matrix(preds_rg_val, Y_test_val.argmax(axis = -1).cpu().numpy()), display_labels=names).plot()
plt.title('xDAWN + RG valence')
plt.show()
ConfusionMatrixDisplay(confusion_matrix(preds_rg_aro, Y_test_aro.argmax(axis = -1).cpu().numpy()), display_labels=names).plot()
plt.title('xDAWN + RG arousal')
plt.show()
ConfusionMatrixDisplay(confusion_matrix(preds_rg_dom, Y_test_dom.argmax(axis = -1).cpu().numpy()), display_labels=names).plot()
plt.title('xDAWN + RG dominance')
plt.show()



