import torch
import numpy as np
import matplotlib.pyplot as plt

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def train_f(model, train_loader, optimizer, loss_fn, device):
    model.train()
    avg_loss = 0
    for batch_index, (X_batch, Y_batch) in enumerate(train_loader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, Y_batch)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
    return avg_loss / len(train_loader)
    


def test_f(model, test_loader, loss_fn, device):
    model.eval()
    correct = 0
    total = 0
    avg_loss = 0
    with torch.no_grad():
        for batch_index, (X_batch, Y_batch) in enumerate(test_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch)
            avg_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            total += Y_batch.size(0)
            _, target = torch.max(Y_batch, 1)
            correct += (predicted == target).sum().item()
    return correct / total, avg_loss / len(test_loader)


def classification_accuracy(preds, Y_test, names, figs_path, selected_emotion, mode):
    acc = np.mean(preds == Y_test)
    print("Subject-dependent classification accuracy on " + selected_emotion + ": %f " % (acc))
    ConfusionMatrixDisplay(confusion_matrix(preds, Y_test), display_labels=names).plot()
    plt.title("Subject-dependent classification accuracy on " + selected_emotion, fontsize=10)
    plt.xlabel("Predicted \n Classification accuracy: %.2f " % (acc))
    plt.tight_layout()
    plt.savefig(figs_path + 'confusion_matrix_subject_' + mode + '_classification_' + selected_emotion +'.png')


def xDawnRG(dataset, n_components, train_indices, test_indices, chans, samples, names, figs_path, info_str):
    # code is taken from PyRiemann's ERP sample script, which is decoding in 
    # the tangent space with a logistic regression
    clf = make_pipeline(XdawnCovariances(n_components),
                        TangentSpace(metric='riemann'),
                        LogisticRegression())
    X_train = dataset.data[train_indices].squeeze().numpy()
    X_test = dataset.data[test_indices].squeeze().numpy()
    Y_train = dataset.targets[train_indices].numpy()
    Y_test = dataset.targets[test_indices].numpy()

    preds_rg = np.zeros(len(Y_test))
    clf.fit(X_train, Y_train.argmax(axis=-1))
    preds_rg = clf.predict(X_test)
    acc = np.mean(preds_rg == Y_test.argmax(axis=-1))

    print("Classification accuracy on " + info_str[:-1] + " with xDawn+RG: %f " % (acc))
    ConfusionMatrixDisplay(confusion_matrix(preds_rg, Y_test.argmax(axis = -1)), display_labels=names).plot()
    plt.title('xDAWN + RG valence')
    plt.xlabel("Predicted \n Classification accuracy: %.2f " % (acc))
    plt.tight_layout()
    plt.savefig(figs_path + 'confusion_matrix_' + info_str + '_xDawnRG.png')