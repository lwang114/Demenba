# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

from collections import defaultdict
import csv
import json
import pandas as pd
import torchaudio
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import random
import os


class AudiosetDataset(Dataset):
    def __init__(
            self,
            audio_dir='/data/sls/d/corpora/original/FHS/FHS_2022_Gold92/all_audios_16kHz/',
            manifest_dir='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs',
            metadata_path='/data/sls/d/fhs/np_dvoice/add_cog_data_(5586)_[11717]_20250131_14_27_9_0179_dvoice_and_npath_add_dr.csv',
            split='train',
            audio_conf={},
            has_review_only=True,
            merge_dur=30,
            n_class=3,
            num_test_per_class=10,
            num_all_per_class=50,
            add_silence=False,
            diarize=True,
            speakers=['Interviewer', 'Participant'],
        ):
        self.split = split
        self.metadata = pd.read_csv(metadata_path)
        if has_review_only:
            self.metadata = self.metadata[self.metadata['has_dementia_review'] > 0]
        print(f'Total number of recordings: {len(self.metadata)}', flush=True)

        manifest_dir = Path(manifest_dir)
        self.split = split
        self.merge_dur = merge_dur
        self.add_silence = add_silence

        self.n_class = n_class
        count = torch.zeros(n_class)
        data_dict = defaultdict(list)
        patients = []
        interviewers = []
        wav2dur = defaultdict(int)
        print(f'Number of classes: {n_class}, {manifest_dir}/{split}_{num_all_per_class*n_class}')
        if os.path.exists(manifest_dir / f'{split}_{num_all_per_class*n_class}.tsv'):
            self.id_dates = []
            wav2lab = {}
            with open(manifest_dir / f'{split}_{num_all_per_class*n_class}.tsv', 'r') as f_tsv,\
                open(manifest_dir / f'{split}_{num_all_per_class*n_class}.lbl', 'r') as f_lbl:
                labels = f_lbl.read().strip().split('\n')
                labels = list(map(int, labels))

                lines = f_tsv.read().strip().split('\n')
                root = lines.pop(0)
                data_dict = defaultdict(list)
                for i, l in tqdm(list(enumerate(lines))):
                    wav_name, start, end = l.strip().split('\t')
                    wav2dur[wav_name] = max(int(end), wav2dur[wav_name])
                    id_date = Path(wav_name).parent.name
                    patient_id = id_date.split('_')[0]
                    interviewer_id = Path(wav_name).stem.split('_')[-1]

                    if not patient_id in patients:
                        patients.append(patient_id)

                    if not interviewer_id in interviewers:
                        interviewers.append(interviewer_id)

                    label = labels[i]

                    if not id_date in self.id_dates:
                        self.id_dates.append(id_date)
                        wav2lab[id_date] = label

                if os.path.exists(manifest_dir / f'{split}_{num_all_per_class*n_class}_diarized_whisper.wrd') and diarize:
                    print(f'Found diarization path {split}_{num_all_per_class*n_class}_diarized_whisper.wrd')
                    with open(manifest_dir / f'{split}_{num_all_per_class*n_class}_diarized_whisper.wrd', 'r') as f:
                        lines = f.read().strip().split('\n')

                for i, l in tqdm(list(enumerate(lines))):
                    wav_name, start, end = l.strip().split('\t')[:3]
                    if len(l.strip().split('\t')) > 3:
                        spk = l.strip().split('\t')[3].split(':')[0].strip()
                        if i < 10:
                            print(spk, spk in speakers)
                    else:
                        spk = 'Participant'

                    if spk in speakers:
                        id_date = Path(wav_name).parent.name
                        label = wav2lab[id_date]

                        start = int(start)
                        end = int(end)

                        wav_path = os.path.join(root, wav_name)
                        data_dict[id_date].append([start, end, [wav_path], label])

            if merge_dur > 0:
                data_dict = self.merge(data_dict, merge_dur) 
            self.wav_list = [x for id_date in self.id_dates for x in data_dict[id_date]]
            new_id_dates = []
            new_labels = []
            for fn in self.wav_list:
                id_date = Path(fn[2][0]).parent.name
                label = fn[-1]
                if not id_date in new_id_dates:
                    new_id_dates.append(id_date)
                    new_labels.append(label)
#            print(f'new_id_dates: {new_id_dates}, new_labels: {new_labels}')  # XXX
            total_dur = sum(list(wav2dur.values())) / (16e3 * 3600) 
            print(f'Number of {split} files: {len(self.wav_list)} from {len(data_dict)} recordings in {total_dur} hours')
            print(f'Total number of patients: {len(patients)}, total number of interviewers: {len(interviewers)}')
        else:
            id_dates = self.metadata['id_date'].tolist()
            self.train_id_dates = []
            self.valid_id_dates = []
            with open(manifest_dir / f'{split}_{num_all_per_class*n_class}.tsv', 'r') as f_tsv,\
                open(manifest_dir / f'{split}_{num_all_per_class*n_class}.tsv', 'w') as f_split,\
                open(manifest_dir / f'{split}_{num_all_per_class*n_class}.lbl', 'w') as f_lbl:
                lines = f_tsv.read().strip().split('\n')
                root = lines.pop(0)
                print(root, file=f_split)
                for i, l in tqdm(list(enumerate(lines))):
                    wav_name, start, end = l.strip().split('\t')
                    id_date = Path(wav_name).parent.name
                    
                    if id_date in id_dates:
                        wav_path = os.path.join(root, wav_name)
                        label = self.get_label(id_date)
                        data_dict[id_date].append([int(start), int(end), [wav_path], label])

                        if count[label] < num_test_per_class:
                            if split in ['valid', 'test']:
                                print('\t'.join([wav_name, start, end]), file=f_split)
                                print(label, file=f_lbl)

                            if not id_date in self.valid_id_dates:
                                count[label] += 1
                                self.valid_id_dates.append(id_date)
                        elif count[label] < num_all_per_class:
                            if split == 'train':
                                print('\t'.join([wav_name, start, end]), file=f_split)
                                print(label, file=f_lbl)

                            if not id_date in self.train_id_dates:
                                count[label] += 1
                                self.train_id_dates.append(id_date)
                        elif count.sum() >= n_class*num_all_per_class:
                            break

            if split == 'train':
                self.id_dates = self.train_id_dates
                data_dict = {id_date:data_dict[id_date] for id_date in self.train_id_dates}
                if merge_dur > 0:
                    data_dict = self.merge(data_dict, merge_dur)
                self.wav_list = [x for id_date in self.train_id_dates for x in data_dict[id_date]]
                print(f'Number of {split} files: {len(self.wav_list)} from {len(self.train_id_dates)} recordings')
            else:
                self.id_dates = self.valid_id_dates
                data_dict = {id_date:data_dict[id_date] for id_date in self.valid_id_dates}
                if merge_dur > 0:
                    data_dict = self.merge(data_dict, merge_dur)
                self.wav_list = [x for id_date in self.valid_id_dates for x in data_dict[id_date]]
                print(f'Number of {split} files: {len(self.wav_list)} from {len(self.valid_id_dates)} recordings')

        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

    def get_label(self, id_date):
        id_dates = self.metadata['id_date'].tolist()
        has_reviews = self.metadata['has_dementia_review'].tolist()

        is_norms = self.metadata['is_norm'].tolist()
        is_mcis = self.metadata['is_mci'].tolist()
        is_dementeds = self.metadata['is_demented'].tolist()

        id_date_idx = id_dates.index(id_date)
        has_review = has_reviews[id_date_idx]
        label = 0
        if has_review:
            is_norm = is_norms[id_date_idx]
            is_mci = is_mcis[id_date_idx]
            is_demented = is_dementeds[id_date_idx]
            if self.n_class == 3:
                if is_norm:
                    label = 0
                elif is_mci:
                    label = 1
                elif is_demented:
                    label = 2
                else:
                    print(f'Warning: possibly corrupt label with is_norm={is_norm}, is_mci={is_mci} and is_demented={is_demented}')
            else:
                label = int(is_norm == 0) 
        return label

    def __getitem__(self, index):
        starts, ends, wav_paths, label = self.wav_list[index]
        id_date = Path(wav_paths[0]).parent.name
        
        prev_end = 0
        wav_path = wav_paths[0]
        x, sr = torchaudio.load(wav_path)
        if self.add_silence:
            start = starts[0]
            end = ends[-1]
            x = x[:, start:end]
        else:
            xs = []
            for start, end in zip(starts, ends):
                x = x[:, start:end]
                xs.append(x)
            x = torch.cat(xs, dim=-1)

        if x.shape[1] < 50:
            print(f'Warning: utterance too short with length {x.shape[1]}')
            x = torch.zeros(1, 16000)
        fbank = torchaudio.compliance.kaldi.fbank(x, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        nframes = fbank.shape[0]

        p = target_length - nframes

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[:target_length]

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.t().unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = fbank.t()

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        onehot_label = F.one_hot(torch.tensor(label), num_classes=self.n_class).float()
        onehot_label[label] = 1.0
        return fbank, onehot_label, id_date

    def __len__(self):
        return len(self.wav_list)

    def merge(self, data_dict, merge_dur):
        new_data_dict = defaultdict(list)
        total = 0
        total_merged = 0
        for id_date in tqdm(list(sorted(data_dict))):
            flist = sorted(data_dict[id_date], key=lambda x:x[0])
            start, end, fns, label = flist[0]
            new_fn = [[start], [end], fns, label]
            new_flist = []
            for start, end, fns, label in flist[1:]:
                total += 1
                if end - new_fn[0][0] > merge_dur * 16e3:
                    total_merged += 1
                    new_flist.append(new_fn)
                    new_fn = [[start], [end], fns, label]
                else:
                    new_fn[0].append(start)
                    new_fn[1].append(end)
                    new_fn[2].extend(fns)

            new_flist.append(new_fn)
            new_data_dict[id_date].extend(new_flist)
            
        print(f'Created {total_merged} merged files from {total} files', flush=True)
        return new_data_dict
