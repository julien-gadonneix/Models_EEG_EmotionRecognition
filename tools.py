import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import torch.distributed as dist
import torch.multiprocessing as mp

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


MODEL_CHOICES = ["EEGNet", "CapsEEGNet", "TCNet"]


def train_f(model, train_loader, optimizer, loss_fn, scaler, device, is_ok, dev1=None):
    model.train()
    avg_loss = 0
    for X_batch, Y_batch in train_loader:
        if model.name not in ['CapsEEGNet', 'TCNet']:
            X_batch, Y_batch = X_batch.to(device=device, memory_format=torch.channels_last), Y_batch.to(device)
        else:
            X_batch = X_batch.to(device)
            if model.name == 'TCNet' and dev1 is not None:
                Y_batch = Y_batch.to(dev1)
            else:
                Y_batch = Y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if is_ok:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, Y_batch)
        else:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch)
        if model.name in ['CapsEEGNet', 'TCNet']:
            loss += torch.norm(model.primaryCaps.caps.weight, p=2) + torch.norm(model.emotionCaps.W, p=2)
        if model.name == 'MLFCapsNet':
            loss += torch.norm(model.primaryCaps.caps.weight, p=2) + torch.norm(model.emotionCaps.W, p=2) + torch.norm(model.conv.weight, p=2)
        avg_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return avg_loss / len(train_loader)


def test_f(model, test_loader, loss_fn, device, is_ok, dev1=None):
    model.eval()
    correct = 0
    total = 0
    avg_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            if model.name not in ['CapsEEGNet', 'TCNet']:
                X_batch, Y_batch = X_batch.to(device=device, memory_format=torch.channels_last), Y_batch.to(device)
            else:
                X_batch = X_batch.to(device)
                if model.name == 'TCNet' and dev1 is not None:
                    Y_batch = Y_batch.to(dev1)
                else:
                    Y_batch = Y_batch.to(device)
            if is_ok:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, Y_batch)
            else:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, Y_batch)
            if model.name in ['CapsEEGNet', 'TCNet']:
                loss += torch.norm(model.primaryCaps.caps.weight, p=2) + torch.norm(model.emotionCaps.W, p=2)
            avg_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            total += Y_batch.size(0)
            target = Y_batch
            correct += (predicted == target).sum().item()
    return correct / total, avg_loss / len(test_loader)


def classification_accuracy(preds, Y_test, names, figs_path, selected_emotion, mode):
    acc = np.mean(preds == Y_test)
    print("Subject-" + mode + " classification accuracy on " + selected_emotion + ": %f " % (acc))
    ConfusionMatrixDisplay(confusion_matrix(preds, Y_test, labels=np.arange(len(names))), display_labels=names).plot()
    plt.title("Subject-" + mode + " classification accuracy on " + selected_emotion, fontsize=10)
    plt.xlabel("Predicted \n Classification accuracy: %.4f " % (acc))
    plt.tight_layout()
    plt.savefig(figs_path + 'confusion_matrix_subject_' + mode + '_classification_' + selected_emotion +'.png')


def draw_loss(losses_train, losses_test, figs_path, selected_emotion, subject):
    plt.figure()
    plt.plot(losses_train, label='Train loss')
    plt.plot(losses_test, label='Test loss')
    plt.title("Loss on " + selected_emotion + " of subject " + subject)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_path + 'loss_subject' + subject + '_' + selected_emotion + '.png')
    plt.close()


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


def margin_loss(y_pred, y_true):
    y_true = F.one_hot(y_true, num_classes=y_pred.shape[1])
    L = y_true * torch.square(torch.maximum(torch.zeros_like(y_pred, device=y_pred.device), 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * torch.square(torch.maximum(torch.zeros_like(y_pred, device=y_pred.device), y_pred - 0.1))
    return torch.mean(torch.sum(L, 1))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Adjust based on available GPUs
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_fn(demo_fn, world_size, args):
    mp.spawn(demo_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)