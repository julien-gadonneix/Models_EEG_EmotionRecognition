import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
from collections import defaultdict


class SEEDDataset(Dataset):
    def __init__(self, path, subjects=None, videos=None, sessions=None, samples=200, start=1, save=False):
        data_path = path + 'data.pt'
        if os.path.exists(data_path):
            print("Loading dataset from file.")
            self.data = torch.load(data_path)
            self.targets = torch.load(path + 'targets.pt')
        else:
            print("Building dataset.")
            self._build(path, subjects, videos, sessions, samples, start, save)


    def _build(self, path, subjects=None, videos=None, sessions=None, samples=200, start=1, save=False):
        wdir = Path(__file__).resolve().parent.parent.parent
        data_path = str(wdir) + '/data/SEED/'

        files_mat = sorted([f for f in os.listdir(data_path) if f.endswith('.mat') and f != 'label.mat'])
        labels_mat = scipy.io.loadmat(data_path + 'label.mat')
        labels = labels_mat['label'][0]

        n_videos = len(labels)

        X = []
        y = []
        if subjects is None:
            print("Dataset with all subjects mixed ...")
            if sessions is None:
                print("... , all sessions mixed ...")
                for file in files_mat:
                    mat = scipy.io.loadmat(data_path + file)
                    if videos is None:
                        if file == files_mat[0]:
                            print("... and all videos mixed.")
                        for j in range(n_videos):
                            key = [k for k in mat if k.endswith(f'_eeg{j+1}')][0]
                            stimuli_eeg_j = mat[key].T
                            stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                            stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                            l = stimuli_eeg_j.shape[0]
                            for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                                X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                                # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                                y.append(labels[j]+1)
                    else:
                        if file == files_mat[0]:
                            print("... and videos:", videos)
                        for video in videos:
                            key = [k for k in mat if k.endswith(f'_eeg{video+1}')][0]
                            stimuli_eeg_j = mat[key].T
                            stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                            stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                            l = stimuli_eeg_j.shape[0]
                            for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                                X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                                # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                                y.append(labels[video]+1)
            else:
                print("... , sessions:", sessions, "...")
                count_map = defaultdict(int)
                filtered_files_mat = []
                for file in files_mat:
                    number = int(file.split('_')[0])
                    if number not in count_map:
                        count_map[number] = 0
                    if count_map[number] in sessions:
                        filtered_files_mat.append(file)
                    count_map[number] += 1
                for file in filtered_files_mat:
                    mat = scipy.io.loadmat(data_path + file)
                    if videos is None:
                        if file == filtered_files_mat[0]:
                            print("... and all videos mixed.")
                        for j in range(n_videos):
                            key = [k for k in mat if k.endswith(f'_eeg{j+1}')][0]
                            stimuli_eeg_j = mat[key].T
                            stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                            stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                            l = stimuli_eeg_j.shape[0]
                            for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                                X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                                # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                                y.append(labels[j]+1)
                    else:
                        if file == filtered_files_mat[0]:
                            print("... and videos:", videos)
                        for video in videos:
                            key = [k for k in mat if k.endswith(f'_eeg{video+1}')][0]
                            stimuli_eeg_j = mat[key].T
                            stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                            stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                            l = stimuli_eeg_j.shape[0]
                            for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                                X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                                # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                                y.append(labels[video]+1)
        else:
            print("Dataset with subjects:", subjects, "...")
            for subject in subjects:
                filtered_files_mat = [f for f in files_mat if f.startswith(f'{subject+1}_')]
                if sessions is None:
                    if subject == subjects[0]:
                        print("... , all sessions mixed ...")
                    for file in filtered_files_mat:
                        mat = scipy.io.loadmat(data_path + file)
                        if videos is None:
                            if subject == subjects[0] and file == filtered_files_mat[0]:
                                print("... and all videos mixed.")
                            for j in range(n_videos):
                                key = [k for k in mat if k.endswith(f'_eeg{j+1}')][0]
                                stimuli_eeg_j = mat[key].T
                                stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                                stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                                l = stimuli_eeg_j.shape[0]
                                for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                                    X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                                    # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                                    y.append(labels[j]+1)
                        else:
                            if subject == subjects[0] and file == filtered_files_mat[0]:
                                print("... and videos:", videos)
                            for video in videos:
                                key = [k for k in mat if k.endswith(f'_eeg{video+1}')][0]
                                stimuli_eeg_j = mat[key].T
                                stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                                stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                                l = stimuli_eeg_j.shape[0]
                                for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                                    X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                                    # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                                    y.append(labels[video]+1)
                else:
                    if subject == subjects[0]:
                        print("... , sessions:", sessions, "...")
                    count_map = defaultdict(int)
                    filtered_filtered_files_mat = []
                    for file in filtered_files_mat:
                        number = int(file.split('_')[0])
                        if number not in count_map:
                            count_map[number] = 0
                        if count_map[number] in sessions:
                            filtered_filtered_files_mat.append(file)
                        count_map[number] += 1
                    for file in filtered_filtered_files_mat:
                        mat = scipy.io.loadmat(data_path + file)
                        if videos is None:
                            if subject == subjects[0] and file == filtered_filtered_files_mat[0]:
                                print("... and all videos mixed.")
                            for j in range(n_videos):
                                key = [k for k in mat if k.endswith(f'_eeg{j+1}')][0]
                                stimuli_eeg_j = mat[key].T
                                stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                                stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                                l = stimuli_eeg_j.shape[0]
                                for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                                    X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                                    # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                                    y.append(labels[j]+1)
                        else:
                            if subject == subjects[0] and file == filtered_filtered_files_mat[0]:
                                print("... and videos:", videos)
                            for video in videos:
                                key = [k for k in mat if k.endswith(f'_eeg{video+1}')][0]
                                stimuli_eeg_j = mat[key].T
                                stimuli_eeg_j -= np.mean(stimuli_eeg_j, axis=0)
                                stimuli_eeg_j /= np.std(stimuli_eeg_j, axis=0)
                                l = stimuli_eeg_j.shape[0]
                                for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                                    X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                                    # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                                    y.append(labels[video]+1)
        X = torch.stack(X)
        self.data = X.unsqueeze(1)
        self.targets = torch.tensor(y, dtype=torch.long)
        if save:
            self._save(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def _save(self, path):
        torch.save(self.data, path + 'data.pt')
        torch.save(self.targets, path + 'targets.pt')
