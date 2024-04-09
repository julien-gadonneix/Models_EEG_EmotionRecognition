import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np


class DREAMERDataset(Dataset):
    def __init__(self, emotion, subject=None, samples=128, start=0):
        data_path = '/users/eleves-a/2021/julien.gadonneix/stage3A/data/DREAMER/'
        mat = scipy.io.loadmat(data_path + 'DREAMER.mat')
        data, eeg_sr, _, eeg_electrodes, n_subjects, n_videos, _, _, _, _  = mat['DREAMER'][0, 0]
        self.samples = samples
        self.start = start
        self.eeg_sr = int(eeg_sr[0, 0])
        eeg_electrodes = eeg_electrodes[0]
        self.eeg_electrodes = [eeg_electrodes[i][0] for i in range(eeg_electrodes.size)]
        self.n_subjects = int(n_subjects[0, 0])
        self.n_videos = int(n_videos[0, 0])

        X = []
        y = []
        if subject is None:
            print("Dataset for subject-independent classification")
            for i in range(self.n_subjects):
                _, _, eeg, _, val, aro, dom = data[0, i][0][0]
                baseline_eeg, stimuli_eeg = eeg[0, 0]
                for j in range(self.n_videos):
                    stimuli_eeg_j = stimuli_eeg[j, 0]
                    baseline_eeg_j = baseline_eeg[j, 0]
                    stimuli_eeg_j -= np.mean(baseline_eeg_j, axis=0)
                    stimuli_eeg_j /= np.std(baseline_eeg_j, axis=0)
                    for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                        l = stimuli_eeg_j.shape[0]
                        X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T * 1000, dtype=torch.float32)) # scale by 1000 due to scaling sensitivity in DL
                        # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T * 1000, dtype=torch.float32)) # scale by 1000 due to scaling sensitivity in DL
                        if emotion == 'valence':
                            y.append(val[j, 0]-1)
                        elif emotion == 'arousal':
                            y.append(aro[j, 0]-1)
                        elif emotion == 'dominance':
                            y.append(dom[j, 0]-1)
                        else:
                            raise ValueError('Invalid emotion')
        else:
            print("Dataset for subject-dependent classification")
            _, _, eeg, _, val, aro, dom = data[0, subject][0][0]
            baseline_eeg, stimuli_eeg = eeg[0, 0]
            for j in range(self.n_videos):
                stimuli_eeg_j = stimuli_eeg[j, 0]
                baseline_eeg_j = baseline_eeg[j, 0]
                stimuli_eeg_j -= np.mean(baseline_eeg_j, axis=0)
                stimuli_eeg_j /= np.std(baseline_eeg_j, axis=0)
                for k in range((stimuli_eeg_j.shape[0]//samples)-start):
                    l = stimuli_eeg_j.shape[0]
                    X.append(torch.tensor(stimuli_eeg_j[l-((k+1)*samples):l-(k*samples), :].T * 1000, dtype=torch.float32)) # scale by 1000 due to scaling sensitivity in DL
                    # X.append(torch.tensor(stimuli_eeg_j[k*samples:(k+1)*samples, :].T * 1000, dtype=torch.float32)) # scale by 1000 due to scaling sensitivity in DL
                    if emotion == 'valence':
                        y.append(val[j, 0]-1)
                    elif emotion == 'arousal':
                        y.append(aro[j, 0]-1)
                    elif emotion == 'dominance':
                        y.append(dom[j, 0]-1)
                    else:
                        raise ValueError('Invalid emotion')
        X = torch.stack(X)
        torch.permute(X, (0, 2, 1))
        self.data = X.unsqueeze(1)
        self.targets = torch.nn.functional.one_hot(torch.tensor(y)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]