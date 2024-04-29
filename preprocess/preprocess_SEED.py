import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
import pandas as pd


class SEEDDataset(Dataset):
    def __init__(self, path, subjects=None, samples=200, start=0, save=False):
        data_path = path + 'data.pt'
        if os.path.exists(data_path):
            print("Loading dataset from file.")
            self.data = torch.load(data_path)
            self.targets = torch.load(path + 'targets.pt')
        else:
            print("Building dataset.")
            self._build(path, subjects, samples, start, save)


    def _build(self, path, subjects=None, samples=200, start=0, save=False):
        wdir = Path(__file__).resolve().parent.parent.parent
        data_path = str(wdir) + '/data/SEED/SEED_EEG/Preprocessed_EEG/'

        # sr = 200
        # n_subjects = 15
        # n_sessions = 3
        n_videos = 15
        n_classes = 3
        files_mat = [f for f in os.listdir(data_path) if f.endswith('.mat') and f != 'label.mat']
        labels_mat = scipy.io.loadmat(data_path + 'label.mat')
        labels = labels_mat['label'][0]

        X = []
        y = []
        if subjects is None:
            print("Dataset with all subjects")
            for file in files_mat:
                mat = scipy.io.loadmat(data_path + file)
                for j in range(n_videos):
                    key = [k for k in mat if k.endswith(f'_eeg{j+1}')][0]
                    stimuli_eeg_j = mat[key].T
                    stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                    stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                    l = stimuli_eeg_j.shape[0]
                    for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                        X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T * 1000, dtype=torch.float32)) # scale by 1000 due to scaling sensitivity in DL
                        # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T * 1000, dtype=torch.float32)) # scale by 1000 due to scaling sensitivity in DL
                        y.append(labels[j]+1)
        else:
            print("Dataset with subject ", subjects)
            for subject in subjects:
                filtered_files_mat = [f for f in files_mat if f.startswith(f'{subject+1}_')]
                for file in filtered_files_mat:
                    mat = scipy.io.loadmat(data_path + file)
                    for j in range(n_videos):
                        key = [k for k in mat if k.endswith(f'_eeg{j+1}')][0]
                        stimuli_eeg_j = mat[key].T
                        stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                        stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                        l = stimuli_eeg_j.shape[0]
                        for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                            X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T * 1000, dtype=torch.float32)) # scale by 1000 due to scaling sensitivity in DL
                            # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T * 1000, dtype=torch.float32)) # scale by 1000 due to scaling sensitivity in DL
                            y.append(labels[j]+1)
        X = torch.stack(X)
        torch.permute(X, (0, 2, 1))
        self.data = X.unsqueeze(1)
        self.targets = torch.nn.functional.one_hot(torch.LongTensor(y), num_classes=n_classes).float()
        if save:
            self._save(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def _save(self, path):
        torch.save(self.data, path + 'data.pt')
        torch.save(self.targets, path + 'targets.pt')
