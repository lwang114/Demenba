# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
from collections import defaultdict
from pathlib import Path
import torchaudio
import numpy as np
import torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import os
import soundfile as sf
from tqdm import tqdm
DOWNSAMPLE_FACTOR=320


def make_data_dict(wav_paths, labels):
    '''
    Create a dictionary that maps wav ids to a list of tuples [start, end, [wav_path], label]
    '''
    data_dict = defaultdict(list)
    for p, y in zip(wav_paths, labels):
        parts = Path(p).stem.split('_')
        spk = '_'.join(parts[:4])
        start, end = parts[-2:]
        start = int(start)
        end = int(end)
        data_dict[spk].append([start, end, [p], y])
    return data_dict


class FHSGoldDataset(Dataset):
    def __init__(
            self, 
            folder='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs_gold92',
            split='all',
            merge_dur=360,
            add_silence=True, 
            merge_dementia_class=True,
        ):
        """
        Dataset that manages audio recordings
        :param dataset_json_file
        """
        self.split = split
        self.add_silence = add_silence
        self.merge_dur = merge_dur
        with open(f'{folder}/all.tsv', 'r') as f_tsv,\
            open(f'{folder}/all.label', 'r') as f_lbl:
            lines = f_tsv.read().strip().split('\n')
            root = lines.pop(0)
            wav_paths = [os.path.join(root, l.strip().split('\t')[0]) for l in lines]
            
            lines = f_lbl.read().strip().split('\n')
            labels = list(map(int, lines))
            if merge_dementia_class:
                labels = list(map(lambda x: int(x > 0), labels))
            self.label_num = len(set(labels))
            self.pos_num = sum(labels)
            self.total_num = len(labels)
            assert(len(labels) == len(wav_paths), "number of wavs and number of labels should match")
            print(f'Number of classes: {self.label_num}')
        
        data_dict = make_data_dict(wav_paths, labels)
        data_dict = self.merge(data_dict, merge_dur)
        self.wav_list = [x+[id_date] for id_date in data_dict for x in data_dict[id_date]]

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        _, _, wav_paths, label, id_date = self.wav_list[index]
        xs = []        
        prev_end = 0
        for wav_path in wav_paths:
            x, sr = torchaudio.load(wav_path)
            if sr != 16000:
                x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=16000)

            if self.add_silence and (self.merge_dur > 0):
                start, end = Path(wav_path).stem.split('_')[-2:]
                start, end = int(start), int(end)

                if prev_end < start:
                    sil = torch.zeros(1, start-prev_end)
                    xs.append(sil)

                prev_end = end
            xs.append(x)

        x = torch.cat(xs, dim=-1)
        if x.shape[-1] > 480000:
            x = x[:, :480000]

        n_pad = DOWNSAMPLE_FACTOR - (x.shape[-1] % DOWNSAMPLE_FACTOR)
        x = F.pad(x, (0, n_pad), value=0)
        size = x.shape[-1] // DOWNSAMPLE_FACTOR
        return {'audio_input': x, 'size': size, 'dementia_label': label, 'wav_id': id_date}

    def __len__(self):
        return len(self.wav_list)

    def collater(self, batch):
        wav_ids = [sample['wav_id'] for sample in batch]
        audios = [sample['audio_input'] for sample in batch]
        sizes = torch.tensor(
            [sample['size'] for sample in batch]
        )
        dementia_labels = torch.tensor(
            [sample['dementia_label'] for sample in batch]
        )
        ids = [sample['wav_id'] for sample in batch]

        max_len = max([x.shape[-1] for x in audios])
        collated_audio = audios[0].new_zeros(len(audios), max_len)
        for i, x in enumerate(audios):
            collated_audio[i, :x.shape[-1]] = x

        return {
            'audio_input': collated_audio,
            'sizes': sizes,
            'dementia_labels': dementia_labels,
            'ids': wav_ids,
        }

    def merge(self, data_dict, merge_dur):
        new_data_dict = defaultdict(list)
        total = 0
        total_merged = 0
        for id_date in tqdm(list(sorted(data_dict))):
            flist = sorted(data_dict[id_date], key=lambda x:x[0])
            start, end, fns, label = flist[0]
            new_fn = [start, end, fns, label]
            new_flist = []
            for start, end, fns, label in flist[1:]:
                total += 1
                if end - new_fn[0] > merge_dur * 16e3:
                    total_merged += 1
                    new_flist.append(new_fn)
                    new_fn = [start, end, fns, label]
                else:
                    new_fn[1] = end
                    new_fn[2].extend(fns)

            new_flist.append(new_fn)
#            print('new_flist[:5]:', new_flist[:5])
            new_data_dict[id_date].extend(new_flist)
            
        print(f'Created {total_merged} merged files from {total} files', flush=True)
        return new_data_dict
