import torch
import numpy as np
import matplotlib.pyplot as plt

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def train(model, epochs, train_loader, validation_loader, optimizer, loss_fn, checkpointer,
          device, figs_path, info_str, save_figs):
    test_losses = []
    train_losses = []
    min_val_loss = np.inf
    for epoch in range(epochs):
        avg_loss = 0.
        model.train()
        it = 0
        for batch_index, (X_batch, Y_batch) in enumerate(train_loader):
            it += 1
            optimizer.zero_grad()
            Y_batch = Y_batch.to(device)
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss /= it
        train_losses.append(avg_loss)

        model.eval()
        it = 0
        with torch.no_grad():
            avg_val_loss = 0.
            avg_val_acc = 0.
            for batch_index, (X_batch, Y_batch) in enumerate(validation_loader):
                it += 1
                Y_batch = Y_batch.to(device)
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                val_loss = loss_fn(y_pred, Y_batch).item()
                avg_val_loss += val_loss
                val_acc = (y_pred.argmax(dim=-1) == Y_batch.argmax(dim=-1)).float().mean().item()
                avg_val_acc += val_acc
            avg_val_loss /= it
            avg_val_acc /= it
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(model.state_dict(), checkpointer)
            test_losses.append(avg_val_loss)
        if epoch % 25 == 0:
            print("Epoch %d: Train loss %.3f, Validation loss %.3f, Validation accuracy %.2f" 
                % (epoch, avg_loss, avg_val_loss, avg_val_acc))
    if save_figs:
        plt.figure()
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend()
        plt.title(info_str[:-1] + '\n' + model.name, fontsize=10)
        plt.tight_layout()
        plt.savefig(figs_path + 'loss_' + info_str + '_' + model.name + '.png')


def test(model, test_loader, names, figs_path, device, info_str, save_figs):
    it = 0
    avg_acc = 0.
    preds_total = []
    Y_test = []
    with torch.no_grad():
        for batch_index, (X_batch, Y_batch) in enumerate(test_loader):
            it += 1
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            probs = model(X_batch).detach().cpu().numpy()
            preds = probs.argmax(axis=-1)
            Y_test.append(Y_batch.argmax(dim=-1).cpu().numpy())
            preds_total.append(preds)
            acc = np.mean(preds == Y_test[-1])
            avg_acc += acc
    avg_acc /= it
    preds = np.concatenate(preds_total)
    Y_test = np.concatenate(Y_test)

    print("Classification accuracy on " + info_str[:-1] + ": %f " % (avg_acc))
    if save_figs:
        ConfusionMatrixDisplay(confusion_matrix(preds, Y_test), display_labels=names).plot()
        plt.title(info_str[:-1] + '\n' + model.name, fontsize=10)
        plt.xlabel("Predicted \n Classification accuracy: %.2f " % (avg_acc))
        plt.tight_layout()
        plt.savefig(figs_path + 'confusion_matrix_' + info_str + '_' + model.name +'.png')
    return preds, Y_test


def subject_dependent_classification_accuracy(preds, Y_test, names, figs_path, selected_emotion):
    acc = np.mean(preds == Y_test)
    print("Subject-dependent classification accuracy on " + selected_emotion + ": %f " % (acc))
    ConfusionMatrixDisplay(confusion_matrix(preds, Y_test), display_labels=names).plot()
    plt.title("Subject-dependent classification accuracy on " + selected_emotion, fontsize=10)
    plt.xlabel("Predicted \n Classification accuracy: %.2f " % (acc))
    plt.tight_layout()
    plt.savefig(figs_path + 'confusion_matrix_subject_dependent_classification_' + selected_emotion +'.png')


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