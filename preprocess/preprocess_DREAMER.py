import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
import mne
import os
from pathlib import Path
from PyEMD import EMD


class DREAMERDataset(Dataset):
    def __init__(self, path, emotion, subjects=None, sessions=None, samples=128, start=1, lowcut=.5, highcut=None, order=3, type="butter",
                 save=False, group_classes=True, tfr=None, use_ecg=False, std=True):
        data_path = path + 'data.pt'
        if os.path.exists(data_path):
            print("Loading dataset from file.")
            self.data = torch.load(data_path)
            self.targets = torch.load(path + 'targets.pt')
            self.class_weights = torch.load(path + 'class_weights.pt')
        else:
            print("Building dataset.")
            self._build(path, emotion, subjects, sessions, samples, start, lowcut, highcut, order, type, save, group_classes, tfr, use_ecg, std)


    def _build(self, path, emotion, subjects=None, sessions=None, samples=128, start=1, lowcut=.5, highcut=None, order=3, type="butter",
               save=False, group_classes=True, tfr=None, use_ecg=False, std=True):
        wdir = Path(__file__).resolve().parent.parent.parent
        data_path = str(wdir) + '/data/DREAMER/'
        mat = scipy.io.loadmat(data_path + 'DREAMER.mat')
        data, eeg_sr, ecg_sr, _, n_subjects, n_videos, _, _, _, _  = mat['DREAMER'][0, 0]
        eeg_sr = int(eeg_sr[0, 0])
        if use_ecg:
            ecg_sr = int(ecg_sr[0, 0])
        n_subjects = int(n_subjects[0, 0])
        n_videos = int(n_videos[0, 0])

        if isinstance(tfr, int):
            emd = EMD()

        X = []
        y = torch.tensor([], dtype=torch.long)
        if subjects is None:
            print("Dataset with all subjects mixed ...")
            for i in range(n_subjects):
                _, _, eeg, ecg, val, aro, dom = data[0, i][0][0]
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
                if use_ecg:
                    baseline_ecg, stimuli_ecg = ecg[0, 0]
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
                        if std:
                            std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
                        stimulis_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)[start:]
                        stimulis_j -= avg_baseline_eeg_j
                        if std:
                            stimulis_j /= std_baseline_eeg_j
                        if use_ecg:
                            stimuli_ecg_j = stimuli_ecg[j, 0].astype(np.float64)
                            baseline_ecg_j = baseline_ecg[j, 0].astype(np.float64)
                            stimuli_ecg_j = mne.filter.filter_data(stimuli_ecg_j.T, ecg_sr, lowcut, highcut, method='iir',
                                                                iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                            baseline_ecg_j = mne.filter.filter_data(baseline_ecg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                                    iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                            stimuli_ecg_j = mne.filter.resample(stimuli_ecg_j, down=ecg_sr/eeg_sr, npad='auto', verbose=False)
                            baseline_ecg_j = mne.filter.resample(baseline_ecg_j, down=ecg_sr/eeg_sr, npad='auto', verbose=False)
                            baselines_ecg_j = baseline_ecg_j.reshape(-1, baseline_ecg_j.shape[0], samples)
                            avg_baseline_ecg_j = baselines_ecg_j.mean(axis=0)
                            if std:
                                std_baseline_ecg_j = baselines_ecg_j.std(axis=0)
                            stimulis_ecg_j = stimuli_ecg_j.reshape(-1, stimuli_ecg_j.shape[0], samples)[start:]
                            stimulis_ecg_j -= avg_baseline_ecg_j
                            if std:
                                stimulis_ecg_j /= std_baseline_ecg_j
                            stimulis_j = np.concatenate((stimulis_j, stimulis_ecg_j), axis=1)
                        if isinstance(tfr, dict):
                            cwt = mne.time_frequency.tfr_array_morlet(stimulis_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                       output=tfr['output'], verbose=False)
                            X.append(torch.tensor(cwt, dtype=torch.float32))
                        elif tfr is not None:
                            imfs = []
                            for i in range(stimulis_j.shape[0]):
                                imfs_ch = []
                                for j in range(stimulis_j.shape[1]):
                                    imf = emd.emd(stimulis_j[i, j], max_imf=tfr)
                                    if imf.shape[0] != tfr+1:
                                        imf = np.concatenate((imf, np.zeros((tfr+1-imf.shape[0], imf.shape[1]))))
                                    imfs_ch.append(imf)
                                imfs.append(imfs_ch)
                            X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                        else:
                            X.append(torch.tensor(stimulis_j, dtype=torch.float32).unsqueeze(1))
                        y = torch.cat((y, torch.tensor(labels[j, 0], dtype=torch.long).repeat(stimulis_j.shape[0])))
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
                        if std:
                            std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
                        stimulis_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)[start:]
                        stimulis_j -= avg_baseline_eeg_j
                        if std:
                            stimulis_j /= std_baseline_eeg_j
                        if use_ecg:
                            stimuli_ecg_j = stimuli_ecg[j, 0].astype(np.float64)
                            baseline_ecg_j = baseline_ecg[j, 0].astype(np.float64)
                            stimuli_ecg_j = mne.filter.filter_data(stimuli_ecg_j.T, ecg_sr, lowcut, highcut, method='iir',
                                                                iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                            baseline_ecg_j = mne.filter.filter_data(baseline_ecg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                                    iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                            stimuli_ecg_j = mne.filter.resample(stimuli_ecg_j, down=ecg_sr/eeg_sr, npad='auto', verbose=False)
                            baseline_ecg_j = mne.filter.resample(baseline_ecg_j, down=ecg_sr/eeg_sr, npad='auto', verbose=False)
                            baselines_ecg_j = baseline_ecg_j.reshape(-1, baseline_ecg_j.shape[0], samples)
                            avg_baseline_ecg_j = baselines_ecg_j.mean(axis=0)
                            if std:
                                std_baseline_ecg_j = baselines_ecg_j.std(axis=0)
                            stimulis_ecg_j = stimuli_ecg_j.reshape(-1, stimuli_ecg_j.shape[0], samples)[start:]
                            stimulis_ecg_j -= avg_baseline_ecg_j
                            if std:
                                stimulis_ecg_j /= std_baseline_ecg_j
                            stimulis_j = np.concatenate((stimulis_j, stimulis_ecg_j), axis=1)
                        if isinstance(tfr, dict):
                            cwt = mne.time_frequency.tfr_array_morlet(stimulis_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                       output=tfr['output'], verbose=False)
                            X.append(torch.tensor(cwt, dtype=torch.float32))
                        elif tfr is not None:
                            imfs = []
                            for i in range(stimulis_j.shape[0]):
                                imfs_ch = []
                                for j in range(stimulis_j.shape[1]):
                                    imf = emd.emd(stimulis_j[i, j], max_imf=tfr)
                                    if imf.shape[0] != tfr+1:
                                        imf = np.concatenate((imf, np.zeros((tfr+1-imf.shape[0], imf.shape[1]))))
                                    imfs_ch.append(imf)
                                imfs.append(imfs_ch)
                            X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                        else:
                            X.append(torch.tensor(stimulis_j, dtype=torch.float32).unsqueeze(1))
                        y = torch.cat((y, torch.tensor(labels[sess, 0], dtype=torch.long).repeat(stimulis_j.shape[0])))
        else:
            print("Dataset with subjects:", subjects, "...")
            for subject in subjects:
                _, _, eeg, ecg, val, aro, dom = data[0, subject][0][0]
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
                if use_ecg:
                    baseline_ecg, stimuli_ecg = ecg[0, 0]
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
                        if std:
                            std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
                        stimulis_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)[start:]
                        stimulis_j -= avg_baseline_eeg_j
                        if std:
                            stimulis_j /= std_baseline_eeg_j
                        if use_ecg:
                            stimuli_ecg_j = stimuli_ecg[j, 0].astype(np.float64)
                            baseline_ecg_j = baseline_ecg[j, 0].astype(np.float64)
                            stimuli_ecg_j = mne.filter.filter_data(stimuli_ecg_j.T, ecg_sr, lowcut, highcut, method='iir',
                                                                iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                            baseline_ecg_j = mne.filter.filter_data(baseline_ecg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                                    iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                            stimuli_ecg_j = mne.filter.resample(stimuli_ecg_j, down=ecg_sr/eeg_sr, npad='auto', verbose=False)
                            baseline_ecg_j = mne.filter.resample(baseline_ecg_j, down=ecg_sr/eeg_sr, npad='auto', verbose=False)
                            baselines_ecg_j = baseline_ecg_j.reshape(-1, baseline_ecg_j.shape[0], samples)
                            avg_baseline_ecg_j = baselines_ecg_j.mean(axis=0)
                            if std:
                                std_baseline_ecg_j = baselines_ecg_j.std(axis=0)
                            stimulis_ecg_j = stimuli_ecg_j.reshape(-1, stimuli_ecg_j.shape[0], samples)[start:]
                            stimulis_ecg_j -= avg_baseline_ecg_j
                            if std:
                                stimulis_ecg_j /= std_baseline_ecg_j
                            stimulis_j = np.concatenate((stimulis_j, stimulis_ecg_j), axis=1)
                        if isinstance(tfr, dict):
                            cwt = mne.time_frequency.tfr_array_morlet(stimulis_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                       output=tfr['output'], verbose=False)
                            X.append(torch.tensor(cwt, dtype=torch.float32))
                        elif tfr is not None:
                            imfs = []
                            for i in range(stimulis_j.shape[0]):
                                imfs_ch = []
                                for j in range(stimulis_j.shape[1]):
                                    imf = emd.emd(stimulis_j[i, j], max_imf=tfr)
                                    if imf.shape[0] != tfr+1:
                                        imf = np.concatenate((imf, np.zeros((tfr+1-imf.shape[0], imf.shape[1]))))
                                    imfs_ch.append(imf)
                                imfs.append(imfs_ch)
                            X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                        else:
                            X.append(torch.tensor(stimulis_j, dtype=torch.float32).unsqueeze(1))
                        y = torch.cat((y, torch.tensor(labels[j, 0], dtype=torch.long).repeat(stimulis_j.shape[0])))
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
                        if std:
                            std_baseline_eeg_j = baselines_eeg_j.std(axis=0)
                        stimulis_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)[start:]
                        stimulis_j -= avg_baseline_eeg_j
                        if std:
                            stimulis_j /= std_baseline_eeg_j
                        if use_ecg:
                            stimuli_ecg_j = stimuli_ecg[j, 0].astype(np.float64)
                            baseline_ecg_j = baseline_ecg[j, 0].astype(np.float64)
                            stimuli_ecg_j = mne.filter.filter_data(stimuli_ecg_j.T, ecg_sr, lowcut, highcut, method='iir',
                                                                iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                            baseline_ecg_j = mne.filter.filter_data(baseline_ecg_j.T, eeg_sr, lowcut, highcut, method='iir',
                                                                    iir_params=dict(order=order, rp=0.1, rs=60, ftype=type), verbose=False)
                            stimuli_ecg_j = mne.filter.resample(stimuli_ecg_j, down=ecg_sr/eeg_sr, npad='auto', verbose=False)
                            baseline_ecg_j = mne.filter.resample(baseline_ecg_j, down=ecg_sr/eeg_sr, npad='auto', verbose=False)
                            baselines_ecg_j = baseline_ecg_j.reshape(-1, baseline_ecg_j.shape[0], samples)
                            avg_baseline_ecg_j = baselines_ecg_j.mean(axis=0)
                            if std:
                                std_baseline_ecg_j = baselines_ecg_j.std(axis=0)
                            stimulis_ecg_j = stimuli_ecg_j.reshape(-1, stimuli_ecg_j.shape[0], samples)[start:]
                            stimulis_ecg_j -= avg_baseline_ecg_j
                            if std:
                                stimulis_ecg_j /= std_baseline_ecg_j
                            stimulis_j = np.concatenate((stimulis_j, stimulis_ecg_j), axis=1)
                        if isinstance(tfr, dict):
                            cwt = mne.time_frequency.tfr_array_morlet(stimulis_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                       output=tfr['output'], verbose=False)
                            X.append(torch.tensor(cwt, dtype=torch.float32))
                        elif tfr is not None:
                            imfs = []
                            for i in range(stimulis_j.shape[0]):
                                imfs_ch = []
                                for j in range(stimulis_j.shape[1]):
                                    imf = emd.emd(stimulis_j[i, j], max_imf=tfr)
                                    if imf.shape[0] != tfr+1:
                                        imf = np.concatenate((imf, np.zeros((tfr+1-imf.shape[0], imf.shape[1]))))
                                    imfs_ch.append(imf)
                                imfs.append(imfs_ch)
                            X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                        else:
                            X.append(torch.tensor(stimulis_j, dtype=torch.float32).unsqueeze(1))
                        y = torch.cat((y, torch.tensor(labels[sess, 0], dtype=torch.long).repeat(stimulis_j.shape[0])))
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
