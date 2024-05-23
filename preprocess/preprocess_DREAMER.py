import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
import mne
import os
from pathlib import Path


class DREAMERDataset(Dataset):
    def __init__(self, path, emotion, subjects=None, sessions=None, samples=128, start=1, lowcut=0.3, highcut=None, order=3, type="butter", save=False):
        data_path = path + 'data.pt'
        if os.path.exists(data_path):
            print("Loading dataset from file.")
            self.data = torch.load(data_path)
            self.targets = torch.load(path + 'targets.pt')
            # self.class_weights = torch.load(path + 'class_weights.pt')
        else:
            print("Building dataset.")
            self._build(path, emotion, subjects, sessions, samples, start, lowcut, highcut, order, type, save)


    def _build(self, path, emotion, subjects=None, sessions=None, samples=128, start=1, lowcut=0.3, highcut=None, order=3, type="butter", save=False):
        wdir = Path(__file__).resolve().parent.parent.parent
        data_path = str(wdir) + '/data/DREAMER/'
        mat = scipy.io.loadmat(data_path + 'DREAMER.mat')
        data, eeg_sr, _, _, n_subjects, n_videos, _, _, _, _  = mat['DREAMER'][0, 0]
        eeg_sr = int(eeg_sr[0, 0])
        n_subjects = int(n_subjects[0, 0])
        n_videos = int(n_videos[0, 0])

        X = []
        y = []
        if subjects is None:
            print("Dataset with all subjects mixed ...")
            for i in range(n_subjects):
                _, _, eeg, _, val, aro, dom = data[0, i][0][0]
                baseline_eeg, stimuli_eeg = eeg[0, 0]
                if sessions is None:
                    if i == 0:
                        print("... and all sessions/videos mixed.")
                    for j in range(n_videos):
                        stimuli_eeg_j = stimuli_eeg[j, 0]
                        baseline_eeg_j = baseline_eeg[j, 0]
                        stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, lowcut, highcut, 
                                            method='iir', 
                                            iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False).T
                        baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, lowcut, highcut, 
                                            method='iir', 
                                            iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False).T
                        stimuli_eeg_j -= np.mean(baseline_eeg_j, axis=0)
                        stimuli_eeg_j /= np.std(baseline_eeg_j, axis=0)
                        l = stimuli_eeg_j.shape[0]
                        for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                            X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                            # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                            if emotion == 'valence':
                                y.append(val[j, 0]-1)
                            elif emotion == 'arousal':
                                y.append(aro[j, 0]-1)
                            elif emotion == 'dominance':
                                y.append(dom[j, 0]-1)
                            else:
                                raise ValueError('Invalid emotion')
                else:
                    if i == 0:
                        print("... and session(s):", sessions)
                    for sess in sessions:
                        stimuli_eeg_j = stimuli_eeg[sess, 0]
                        baseline_eeg_j = baseline_eeg[sess, 0]
                        stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, lowcut, highcut, 
                                            method='iir', 
                                            iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False).T
                        baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, lowcut, highcut, 
                                            method='iir', 
                                            iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False).T
                        stimuli_eeg_j -= np.mean(baseline_eeg_j, axis=0)
                        stimuli_eeg_j /= np.std(baseline_eeg_j, axis=0)
                        l = stimuli_eeg_j.shape[0]
                        for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                            X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                            # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                            if emotion == 'valence':
                                y.append(val[sess, 0]-1)
                            elif emotion == 'arousal':
                                y.append(aro[sess, 0]-1)
                            elif emotion == 'dominance':
                                y.append(dom[sess, 0]-1)
                            else:
                                raise ValueError('Invalid emotion')
        else:
            print("Dataset with subjects:", subjects, "...")
            for subject in subjects:
                _, _, eeg, _, val, aro, dom = data[0, subject][0][0]
                baseline_eeg, stimuli_eeg = eeg[0, 0]
                if sessions is None:
                    if subject == subjects[0]:
                        print("... and all sessions/videos mixed.")
                    for j in range(n_videos):
                        stimuli_eeg_j = stimuli_eeg[j, 0]
                        baseline_eeg_j = baseline_eeg[j, 0]
                        stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, lowcut, highcut, 
                                            method='iir', 
                                            iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False).T
                        baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, lowcut, highcut, 
                                            method='iir', 
                                            iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False).T
                        stimuli_eeg_j -= np.mean(baseline_eeg_j, axis=0)
                        stimuli_eeg_j /= np.std(baseline_eeg_j, axis=0)
                        l = stimuli_eeg_j.shape[0]
                        for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                            X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                            # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                            if emotion == 'valence':
                                y.append(val[j, 0]-1)
                            elif emotion == 'arousal':
                                y.append(aro[j, 0]-1)
                            elif emotion == 'dominance':
                                y.append(dom[j, 0]-1)
                            else:
                                raise ValueError('Invalid emotion')
                else:
                    if subject == subjects[0]:
                        print("... and session(s):", sessions)
                    for sess in sessions:
                        stimuli_eeg_j = stimuli_eeg[sess, 0]
                        baseline_eeg_j = baseline_eeg[sess, 0]
                        stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, lowcut, highcut, 
                                            method='iir', 
                                            iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False).T
                        baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, lowcut, highcut, 
                                            method='iir', 
                                            iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False).T
                        stimuli_eeg_j -= np.mean(baseline_eeg_j, axis=0)
                        stimuli_eeg_j /= np.std(baseline_eeg_j, axis=0)
                        l = stimuli_eeg_j.shape[0]
                        for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                            X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T, dtype=torch.float32))
                            # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T, dtype=torch.float32))
                            if emotion == 'valence':
                                y.append(val[sess, 0]-1)
                            elif emotion == 'arousal':
                                y.append(aro[sess, 0]-1)
                            elif emotion == 'dominance':
                                y.append(dom[sess, 0]-1)
                            else:
                                raise ValueError('Invalid emotion')
        X = torch.stack(X)
        self.data = X.unsqueeze(1)
        self.targets = torch.tensor(y, dtype=torch.long)
        # self.class_weights = torch.tensor(1. / self.targets.mean(dim=0)) TODO: Implement class weights
        if save:
            self._save(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def _save(self, path):
        torch.save(self.data, path + 'data.pt')
        torch.save(self.targets, path + 'targets.pt')
        # torch.save(self.class_weights, path + 'class_weights.pt')
