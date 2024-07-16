import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import mne
from PyEMD import EMD, EEMD, CEEMDAN


class SEEDDataset(Dataset):
    def __init__(self, path, subjects=None, videos=None, sessions=None, samples=200, start=1, save=False, tfr=None, std=True, n_jobs=1):
        data_path = path + 'data.pt'
        if os.path.exists(data_path):
            print("Loading dataset from file.")
            self.data = torch.load(data_path)
            self.targets = torch.load(path + 'targets.pt')
        else:
            print("Building dataset.")
            self._build(path, subjects, videos, sessions, samples, start, save, tfr, std, n_jobs)


    def _build(self, path, subjects=None, videos=None, sessions=None, samples=200, start=1, save=False, tfr=None, std=True, n_jobs=1):
        wdir = Path(__file__).resolve().parent.parent.parent
        data_path = str(wdir) + '/data/SEED/'
        files_mat = sorted([f for f in os.listdir(data_path) if f.endswith('.mat') and f != 'label.mat'])
        labels_mat = scipy.io.loadmat(data_path + 'label.mat')
        labels = labels_mat['label'][0]
        n_videos = len(labels)
        eeg_sr = 200

        if tfr.keys() == {'emd'}:
            emd = EMD()
        elif tfr.keys() == {'eemd', 'sep_trends'}:
            eemd = EEMD(trials=2, parallel=True, processes=n_jobs, separate_trends=tfr['sep_trends'])
        elif tfr.keys() == {'ceemdan', 'beta_prog'}:
            ceemdan = CEEMDAN(trials=2, parallel=True, processes=n_jobs, beta_progress=tfr['beta_prog'])

        X = []
        y = torch.tensor([], dtype=torch.long)
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
                            stimuli_eeg_j = mat[key][:, 1:]
                            stimuli_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)
                            avg_stimuli_eeg_j = np.mean(stimuli_eeg_j, axis=0)
                            if std:
                                std_stimuli_eeg_j = np.std(stimuli_eeg_j, axis=0)
                            stimuli_eeg_j -= avg_stimuli_eeg_j
                            if std:
                                stimuli_eeg_j /= std_stimuli_eeg_j
                            if tfr is not None:
                                if tfr.keys() == {'freqs', 'output'}:
                                    cwt = mne.time_frequency.tfr_array_morlet(stimuli_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                            output=tfr['output'], n_jobs=n_jobs, verbose=False)
                                    X.append(torch.tensor(cwt, dtype=torch.float32))
                                elif tfr.keys() == {'emd'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = emd.emd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['emd'])
                                            if imf.shape[0] != tfr['emd']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['emd']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                elif tfr.keys() == {'eemd', 'sep_trends'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = eemd.eemd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['eemd'])
                                            if imf.shape[0] != tfr['eemd']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['eemd']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                elif tfr.keys() == {'ceemdan', 'beta_prog'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = ceemdan.ceemdan(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['ceemdan'])
                                            if imf.shape[0] != tfr['ceemdan']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['ceemdan']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                            else:
                                X.append(torch.tensor(stimuli_eeg_j, dtype=torch.float32).unsqueeze(1))
                            y = torch.cat((y, torch.tensor(labels[j]+1, dtype=torch.long).repeat(stimuli_eeg_j.shape[0])))
                    else:
                        if file == files_mat[0]:
                            print("... and videos:", videos)
                        for video in videos:
                            key = [k for k in mat if k.endswith(f'_eeg{video+1}')][0]
                            stimuli_eeg_j = mat[key][:, 1:]
                            stimuli_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)
                            avg_stimuli_eeg_j = np.mean(stimuli_eeg_j, axis=0)
                            if std:
                                std_stimuli_eeg_j = np.std(stimuli_eeg_j, axis=0)
                            stimuli_eeg_j -= avg_stimuli_eeg_j
                            if std:
                                stimuli_eeg_j /= std_stimuli_eeg_j
                            if tfr is not None:
                                if tfr.keys() == {'freqs', 'output'}:
                                    cwt = mne.time_frequency.tfr_array_morlet(stimuli_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                            output=tfr['output'], n_jobs=n_jobs, verbose=False)
                                    X.append(torch.tensor(cwt, dtype=torch.float32))
                                elif tfr.keys() == {'emd'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = emd.emd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['emd'])
                                            if imf.shape[0] != tfr['emd']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['emd']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                elif tfr.keys() == {'eemd', 'sep_trends'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = eemd.eemd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['eemd'])
                                            if imf.shape[0] != tfr['eemd']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['eemd']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                elif tfr.keys() == {'ceemdan', 'beta_prog'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = ceemdan.ceemdan(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['ceemdan'])
                                            if imf.shape[0] != tfr['ceemdan']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['ceemdan']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                            else:
                                X.append(torch.tensor(stimuli_eeg_j, dtype=torch.float32).unsqueeze(1))
                            y = torch.cat((y, torch.tensor(labels[video]+1, dtype=torch.long).repeat(stimuli_eeg_j.shape[0])))
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
                            stimuli_eeg_j = mat[key][:, 1:]
                            stimuli_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)
                            avg_stimuli_eeg_j = np.mean(stimuli_eeg_j, axis=0)
                            if std:
                                std_stimuli_eeg_j = np.std(stimuli_eeg_j, axis=0)
                            stimuli_eeg_j -= avg_stimuli_eeg_j
                            if std:
                                stimuli_eeg_j /= std_stimuli_eeg_j
                            if tfr is not None:
                                if tfr.keys() == {'freqs', 'output'}:
                                    cwt = mne.time_frequency.tfr_array_morlet(stimuli_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                            output=tfr['output'], n_jobs=n_jobs, verbose=False)
                                    X.append(torch.tensor(cwt, dtype=torch.float32))
                                elif tfr.keys() == {'emd'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = emd.emd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['emd'])
                                            if imf.shape[0] != tfr['emd']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['emd']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                elif tfr.keys() == {'eemd', 'sep_trends'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = eemd.eemd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['eemd'])
                                            if imf.shape[0] != tfr['eemd']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['eemd']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                elif tfr.keys() == {'ceemdan', 'beta_prog'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = ceemdan.ceemdan(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['ceemdan'])
                                            if imf.shape[0] != tfr['ceemdan']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['ceemdan']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                            else:
                                X.append(torch.tensor(stimuli_eeg_j, dtype=torch.float32).unsqueeze(1))
                            y = torch.cat((y, torch.tensor(labels[j]+1, dtype=torch.long).repeat(stimuli_eeg_j.shape[0])))
                    else:
                        if file == filtered_files_mat[0]:
                            print("... and videos:", videos)
                        for video in videos:
                            key = [k for k in mat if k.endswith(f'_eeg{video+1}')][0]
                            stimuli_eeg_j = mat[key][:, 1:]
                            stimuli_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)
                            avg_stimuli_eeg_j = np.mean(stimuli_eeg_j, axis=0)
                            if std:
                                std_stimuli_eeg_j = np.std(stimuli_eeg_j, axis=0)
                            stimuli_eeg_j -= avg_stimuli_eeg_j
                            if std:
                                stimuli_eeg_j /= std_stimuli_eeg_j
                            if tfr is not None:
                                if tfr.keys() == {'freqs', 'output'}:
                                    cwt = mne.time_frequency.tfr_array_morlet(stimuli_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                            output=tfr['output'], n_jobs=n_jobs, verbose=False)
                                    X.append(torch.tensor(cwt, dtype=torch.float32))
                                elif tfr.keys() == {'emd'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = emd.emd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['emd'])
                                            if imf.shape[0] != tfr['emd']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['emd']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                elif tfr.keys() == {'eemd', 'sep_trends'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = eemd.eemd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['eemd'])
                                            if imf.shape[0] != tfr['eemd']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['eemd']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                elif tfr.keys() == {'ceemdan', 'beta_prog'}:
                                    imfs = []
                                    for i_emd in range(stimuli_eeg_j.shape[0]):
                                        imfs_ch = []
                                        for j_emd in range(stimuli_eeg_j.shape[1]):
                                            imf = ceemdan.ceemdan(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['ceemdan'])
                                            if imf.shape[0] != tfr['ceemdan']+1:
                                                imf = np.concatenate((imf, np.zeros((tfr['ceemdan']+1-imf.shape[0], imf.shape[1]))))
                                            imfs_ch.append(imf)
                                        imfs.append(imfs_ch)
                                    X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                            else:
                                X.append(torch.tensor(stimuli_eeg_j, dtype=torch.float32).unsqueeze(1))
                            y = torch.cat((y, torch.tensor(labels[video]+1, dtype=torch.long).repeat(stimuli_eeg_j.shape[0])))
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
                                stimuli_eeg_j = mat[key][:, 1:]
                                stimuli_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)
                                avg_stimuli_eeg_j = np.mean(stimuli_eeg_j, axis=0)
                                if std:
                                    std_stimuli_eeg_j = np.std(stimuli_eeg_j, axis=0)
                                stimuli_eeg_j -= avg_stimuli_eeg_j
                                if std:
                                    stimuli_eeg_j /= std_stimuli_eeg_j
                                if tfr is not None:
                                    if tfr.keys() == {'freqs', 'output'}:
                                        cwt = mne.time_frequency.tfr_array_morlet(stimuli_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                                output=tfr['output'], n_jobs=n_jobs, verbose=False)
                                        X.append(torch.tensor(cwt, dtype=torch.float32))
                                    elif tfr.keys() == {'emd'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = emd.emd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['emd'])
                                                if imf.shape[0] != tfr['emd']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['emd']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                    elif tfr.keys() == {'eemd', 'sep_trends'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = eemd.eemd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['eemd'])
                                                if imf.shape[0] != tfr['eemd']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['eemd']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                    elif tfr.keys() == {'ceemdan', 'beta_prog'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = ceemdan.ceemdan(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['ceemdan'])
                                                if imf.shape[0] != tfr['ceemdan']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['ceemdan']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                else:
                                    X.append(torch.tensor(stimuli_eeg_j, dtype=torch.float32).unsqueeze(1))
                                y = torch.cat((y, torch.tensor(labels[j]+1, dtype=torch.long).repeat(stimuli_eeg_j.shape[0])))
                        else:
                            if subject == subjects[0] and file == filtered_files_mat[0]:
                                print("... and videos:", videos)
                            for video in videos:
                                key = [k for k in mat if k.endswith(f'_eeg{video+1}')][0]
                                stimuli_eeg_j = mat[key][:, 1:]
                                stimuli_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)
                                avg_stimuli_eeg_j = np.mean(stimuli_eeg_j, axis=0)
                                if std:
                                    std_stimuli_eeg_j = np.std(stimuli_eeg_j, axis=0)
                                stimuli_eeg_j -= avg_stimuli_eeg_j
                                if std:
                                    stimuli_eeg_j /= std_stimuli_eeg_j
                                if tfr is not None:
                                    if tfr.keys() == {'freqs', 'output'}:
                                        cwt = mne.time_frequency.tfr_array_morlet(stimuli_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                                output=tfr['output'], n_jobs=n_jobs, verbose=False)
                                        X.append(torch.tensor(cwt, dtype=torch.float32))
                                    elif tfr.keys() == {'emd'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = emd.emd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['emd'])
                                                if imf.shape[0] != tfr['emd']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['emd']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                    elif tfr.keys() == {'eemd', 'sep_trends'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = eemd.eemd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['eemd'])
                                                if imf.shape[0] != tfr['eemd']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['eemd']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                    elif tfr.keys() == {'ceemdan', 'beta_prog'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = ceemdan.ceemdan(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['ceemdan'])
                                                if imf.shape[0] != tfr['ceemdan']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['ceemdan']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                else:
                                    X.append(torch.tensor(stimuli_eeg_j, dtype=torch.float32).unsqueeze(1))
                                y = torch.cat((y, torch.tensor(labels[video]+1, dtype=torch.long).repeat(stimuli_eeg_j.shape[0])))
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
                                stimuli_eeg_j = mat[key][:, 1:]
                                stimuli_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)
                                avg_stimuli_eeg_j = np.mean(stimuli_eeg_j, axis=0)
                                if std:
                                    std_stimuli_eeg_j = np.std(stimuli_eeg_j, axis=0)
                                stimuli_eeg_j -= avg_stimuli_eeg_j
                                if std:
                                    stimuli_eeg_j /= std_stimuli_eeg_j
                                if tfr is not None:
                                    if tfr.keys() == {'freqs', 'output'}:
                                        cwt = mne.time_frequency.tfr_array_morlet(stimuli_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                                output=tfr['output'], n_jobs=n_jobs, verbose=False)
                                        X.append(torch.tensor(cwt, dtype=torch.float32))
                                    elif tfr.keys() == {'emd'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = emd.emd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['emd'])
                                                if imf.shape[0] != tfr['emd']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['emd']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                    elif tfr.keys() == {'eemd', 'sep_trends'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = eemd.eemd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['eemd'])
                                                if imf.shape[0] != tfr['eemd']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['eemd']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                    elif tfr.keys() == {'ceemdan', 'beta_prog'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = ceemdan.ceemdan(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['ceemdan'])
                                                if imf.shape[0] != tfr['ceemdan']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['ceemdan']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                else:
                                    X.append(torch.tensor(stimuli_eeg_j, dtype=torch.float32).unsqueeze(1))
                                y = torch.cat((y, torch.tensor(labels[j]+1, dtype=torch.long).repeat(stimuli_eeg_j.shape[0])))
                        else:
                            if subject == subjects[0] and file == filtered_filtered_files_mat[0]:
                                print("... and videos:", videos)
                            for video in videos:
                                key = [k for k in mat if k.endswith(f'_eeg{video+1}')][0]
                                stimuli_eeg_j = mat[key][:, 1:]
                                stimuli_eeg_j = stimuli_eeg_j.reshape(-1, stimuli_eeg_j.shape[0], samples)
                                avg_stimuli_eeg_j = np.mean(stimuli_eeg_j, axis=0)
                                if std:
                                    std_stimuli_eeg_j = np.std(stimuli_eeg_j, axis=0)
                                stimuli_eeg_j -= avg_stimuli_eeg_j
                                if std:
                                    stimuli_eeg_j /= std_stimuli_eeg_j
                                if tfr is not None:
                                    if tfr.keys() == {'freqs', 'output'}:
                                        cwt = mne.time_frequency.tfr_array_morlet(stimuli_eeg_j, eeg_sr, tfr['freqs'], n_cycles=tfr['freqs']/2., zero_mean=True,
                                                                                output=tfr['output'], n_jobs=n_jobs, verbose=False)
                                        X.append(torch.tensor(cwt, dtype=torch.float32))
                                    elif tfr.keys() == {'emd'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = emd.emd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['emd'])
                                                if imf.shape[0] != tfr['emd']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['emd']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                    elif tfr.keys() == {'eemd', 'sep_trends'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = eemd.eemd(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['eemd'])
                                                if imf.shape[0] != tfr['eemd']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['eemd']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                    elif tfr.keys() == {'ceemdan', 'beta_prog'}:
                                        imfs = []
                                        for i_emd in range(stimuli_eeg_j.shape[0]):
                                            imfs_ch = []
                                            for j_emd in range(stimuli_eeg_j.shape[1]):
                                                imf = ceemdan.ceemdan(stimuli_eeg_j[i_emd, j_emd], max_imf=tfr['ceemdan'])
                                                if imf.shape[0] != tfr['ceemdan']+1:
                                                    imf = np.concatenate((imf, np.zeros((tfr['ceemdan']+1-imf.shape[0], imf.shape[1]))))
                                                imfs_ch.append(imf)
                                            imfs.append(imfs_ch)
                                        X.append(torch.tensor(np.array(imfs), dtype=torch.float32))
                                else:
                                    X.append(torch.tensor(stimuli_eeg_j, dtype=torch.float32).unsqueeze(1))
                                y = torch.cat((y, torch.tensor(labels[video]+1, dtype=torch.long).repeat(stimuli_eeg_j.shape[0])))
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
