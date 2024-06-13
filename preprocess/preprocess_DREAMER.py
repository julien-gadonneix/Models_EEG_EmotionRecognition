import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
import mne
import os
from pathlib import Path


class DREAMERDataset(Dataset):
    def __init__(self, path, emotion, subjects=None, sessions=None, samples=128, start=1, lowcut=0.3, highcut=None, order=3, type="butter", save=False, group_classes=True, tfr=None):
        data_path = path + 'data.pt'
        if os.path.exists(data_path):
            print("Loading dataset from file.")
            self.data = torch.load(data_path)
            self.targets = torch.load(path + 'targets.pt')
            self.class_weights = torch.load(path + 'class_weights.pt')
        else:
            print("Building dataset.")
            self._build(path, emotion, subjects, sessions, samples, start, lowcut, highcut, order, type, save, group_classes, tfr)


    def _build(self, path, emotion, subjects=None, sessions=None, samples=128, start=1, lowcut=0.3, highcut=None, order=3, type="butter", save=False, group_classes=True, tfr=None):
        wdir = Path(__file__).resolve().parent.parent.parent
        data_path = str(wdir) + '/data/DREAMER/'
        mat = scipy.io.loadmat(data_path + 'DREAMER.mat')
        data, eeg_sr, _, _, n_subjects, n_videos, _, _, _, _  = mat['DREAMER'][0, 0]
        eeg_sr = int(eeg_sr[0, 0])
        n_subjects = int(n_subjects[0, 0])
        n_videos = int(n_videos[0, 0])

        X = []
        y = torch.tensor([], dtype=torch.long)
        if subjects is None:
            print("Dataset with all subjects mixed ...")
            for i in range(n_subjects):
                _, _, eeg, _, val, aro, dom = data[0, i][0][0]
                if emotion == 'valence':
                    if group_classes:
                        labels = (val >= 3) * 1
                    else:
                        labels = val - 1
                elif emotion == 'arousal':
                    if group_classes:
                        labels = (aro >= 3) * 1
                    else:
                        labels = aro - 1
                elif emotion == 'dominance':
                    if group_classes:
                        labels = (dom >= 3) * 1
                    else:
                        labels = dom - 1
                else:
                    raise ValueError('Invalid emotion')
                baseline_eeg, stimuli_eeg = eeg[0, 0]
                if sessions is None:
                    if i == 0:
                        print("... and all sessions/videos mixed.")
                    for j in range(n_videos):
                        stimuli_eeg_j = stimuli_eeg[j, 0]
                        baseline_eeg_j = baseline_eeg[j, 0]
                        stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                               iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                        baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                                iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                        baselines_eeg_j = baseline_eeg_j.reshape(-1, baseline_eeg_j.shape[0], samples)
                        avg_baseline_eeg_j = baselines_eeg_j.mean(axis=0)
                        std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
                        stimulis_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)[start:]
                        stimulis_eeg_j -= avg_baseline_eeg_j
                        stimulis_eeg_j /= std_baseline_eeg_j
                        if tfr is not None:
                            cwt = mne.time_frequency.tfr_array_morlet(stimulis_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                       output=tfr['output'], verbose=False)
                            X.append(torch.tensor(cwt, dtype=torch.float32))
                            y = torch.cat((y, torch.tensor(labels[j, 0], dtype=torch.long).repeat(cwt.shape[0])))
                        else:
                            X.append(torch.tensor(stimulis_eeg_j, dtype=torch.float32).unsqueeze(1))
                            y = torch.cat((y, torch.tensor(labels[j, 0], dtype=torch.long).repeat(stimulis_eeg_j.shape[0])))
                else:
                    if i == 0:
                        print("... and session(s):", sessions)
                    for sess in sessions:
                        stimuli_eeg_j = stimuli_eeg[sess, 0]
                        baseline_eeg_j = baseline_eeg[sess, 0]
                        stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                               iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                        baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                                iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                        baselines_eeg_j = baseline_eeg_j.reshape(-1, baseline_eeg_j.shape[0], samples)
                        avg_baseline_eeg_j = baselines_eeg_j.mean(axis=0)
                        std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
                        stimulis_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)[start:]
                        stimulis_eeg_j -= avg_baseline_eeg_j
                        stimulis_eeg_j /= std_baseline_eeg_j
                        if tfr is not None:
                            cwt = mne.time_frequency.tfr_array_morlet(stimulis_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                       output=tfr['output'], verbose=False)
                            X.append(torch.tensor(cwt, dtype=torch.float32))
                            y = torch.cat((y, torch.tensor(labels[sess, 0], dtype=torch.long).repeat(cwt.shape[0])))
                        else:
                            X.append(torch.tensor(stimulis_eeg_j, dtype=torch.float32).unsqueeze(1))
                            y = torch.cat((y, torch.tensor(labels[sess, 0], dtype=torch.long).repeat(stimulis_eeg_j.shape[0])))
        else:
            print("Dataset with subjects:", subjects, "...")
            for subject in subjects:
                _, _, eeg, _, val, aro, dom = data[0, subject][0][0]
                if emotion == 'valence':
                    if group_classes:
                        labels = (val >= 3) * 1
                    else:
                        labels = val - 1
                elif emotion == 'arousal':
                    if group_classes:
                        labels = (aro >= 3) * 1
                    else:
                        labels = aro - 1
                elif emotion == 'dominance':
                    if group_classes:
                        labels = (dom >= 3) * 1
                    else:
                        labels = dom - 1
                else:
                    raise ValueError('Invalid emotion')
                baseline_eeg, stimuli_eeg = eeg[0, 0]
                if sessions is None:
                    if subject == subjects[0]:
                        print("... and all sessions/videos mixed.")
                    for j in range(n_videos):
                        stimuli_eeg_j = stimuli_eeg[j, 0]
                        baseline_eeg_j = baseline_eeg[j, 0]
                        stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                               iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                        baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                                iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                        baselines_eeg_j = baseline_eeg_j.reshape(-1, baseline_eeg_j.shape[0], samples)
                        avg_baseline_eeg_j = baselines_eeg_j.mean(axis=0)
                        std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
                        stimulis_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)[start:]
                        stimulis_eeg_j -= avg_baseline_eeg_j
                        stimulis_eeg_j /= std_baseline_eeg_j
                        if tfr is not None:
                            cwt = mne.time_frequency.tfr_array_morlet(stimulis_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                       output=tfr['output'], verbose=False)
                            X.append(torch.tensor(cwt, dtype=torch.float32))
                            y = torch.cat((y, torch.tensor(labels[j, 0], dtype=torch.long).repeat(cwt.shape[0])))
                        else:
                            X.append(torch.tensor(stimulis_eeg_j, dtype=torch.float32).unsqueeze(1))
                            y = torch.cat((y, torch.tensor(labels[j, 0], dtype=torch.long).repeat(stimulis_eeg_j.shape[0])))
                else:
                    if subject == subjects[0]:
                        print("... and session(s):", sessions)
                    for sess in sessions:
                        stimuli_eeg_j = stimuli_eeg[sess, 0]
                        baseline_eeg_j = baseline_eeg[sess, 0]
                        stimuli_eeg_j = mne.filter.filter_data(stimuli_eeg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                               iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                        baseline_eeg_j = mne.filter.filter_data(baseline_eeg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                                iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                        baselines_eeg_j = baseline_eeg_j.reshape(-1, baseline_eeg_j.shape[0], samples)
                        avg_baseline_eeg_j = baselines_eeg_j.mean(axis=0)
                        std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
                        stimulis_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)[start:]
                        stimulis_eeg_j -= avg_baseline_eeg_j
                        stimulis_eeg_j /= std_baseline_eeg_j
                        if tfr is not None:
                            cwt = mne.time_frequency.tfr_array_morlet(stimulis_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                       output=tfr['output'], verbose=False)
                            X.append(torch.tensor(cwt, dtype=torch.float32))
                            y = torch.cat((y, torch.tensor(labels[sess, 0], dtype=torch.long).repeat(cwt.shape[0])))
                        else:
                            X.append(torch.tensor(stimulis_eeg_j, dtype=torch.float32).unsqueeze(1))
                            y = torch.cat((y, torch.tensor(labels[sess, 0], dtype=torch.long).repeat(stimulis_eeg_j.shape[0])))
        X = torch.cat(X, dim=0)
        self.data = X
        self.targets = y
        nb_classes = 2 if group_classes else 5
        all_classes = torch.arange(nb_classes)
        unique_classes, counts = torch.unique(self.targets, return_counts=True)
        all_counts = torch.zeros_like(all_classes, dtype=torch.float)
        for i, cls in enumerate(all_classes):
            if cls in unique_classes:
                all_counts[i] = counts[unique_classes == cls].item()
        self.class_weights = all_counts.float() / counts.sum()
        if save:
            self._save(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def _save(self, path):
        torch.save(self.data, path + 'data.pt')
        torch.save(self.targets, path + 'targets.pt')
        torch.save(self.class_weights, path + 'class_weights.pt')
