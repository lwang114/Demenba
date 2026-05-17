from collections import defaultdict
import csv
import json
import pandas as pd
import torchaudio
import numpy as np
import torch
import torchaudio
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
import random
import os

def make_data_dict(texts, wav_paths, labels):
    data_dict = defaultdict(list)
    wav_ids = []
    for text, p, y in zip(texts, wav_paths, labels):
        parts = Path(p).stem.split('_')
        wav_id = '_'.join(parts[:4])
        if not wav_id in wav_ids:
            wav_ids.append(wav_id)
        start, end = parts[-2:]
        start = int(start)
        end = int(end)
        data_dict[wav_id].append([start, end, text, y])
    return data_dict, wav_ids 


def merge(data_dict, merge_dur):
    new_data_dict = defaultdict(list)
    total = 0
    total_merged = 0
    for id_date in tqdm(list(sorted(data_dict))):
        flist = sorted(data_dict[id_date], key=lambda x:x[0])
        start, end, text, label = flist[0]
        new_fn = [start, end, text, label]
        new_flist = []
        for start, end, text, label in flist[1:]:
            total += 1
            if end - new_fn[0] > merge_dur * 16e3:
                total_merged += 1
                new_flist.append(new_fn)
                new_fn = [start, end, text, label]
            else:
                new_fn[1] = end
                new_fn[2] += f'\n{text}'

        new_flist.append(new_fn)
        new_data_dict[id_date].extend(new_flist)
        
    print(f'Created {total_merged} merged files from {total} files', flush=True)
    return new_data_dict


class FHSGoldTextDataset(Dataset):
    def __init__(
            self,
            manifest_dir='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs',
            split='train',
            merge_dur=30,
            n_class=3,
            feat_type='bert-base-cased',
            text_type='raw',
        ):
        if n_class == 1:
            n_class = 2
        self.split = split
        self.merge_dur = merge_dur
        self.tokenizer = AutoTokenizer.from_pretrained(feat_type, use_fast=True)

        manifest_dir = Path(manifest_dir)
        with open(manifest_dir / 'all.tsv', 'r') as f_tsv,\
            open(manifest_dir / 'all.label', 'r') as f_lbl:
            lines = f_tsv.read().strip().split('\n')
            root = lines.pop(0)
            wav_paths = [os.path.join(root, l.strip().split('\t')[0]) for l in lines]
            
            lines = f_lbl.read().strip().split('\n')
            labels = list(map(int, lines))
            if n_class == 2:
                labels = list(map(lambda x: int(x > 0), labels))
           
            self.label_num = len(set(labels))
            self.pos_num = sum(labels)
            self.total_num = len(labels)
            assert len(labels) == len(wav_paths), "number of wavs and number of labels should match"
            print(f'Number of classes: {self.label_num}')
        
        self.n_class = n_class if n_class > 1 else 2

        with open(manifest_dir / f'all_{text_type}.wrd', 'r') as f:
            texts = f.read().strip().split('\n')
            wav2seg, fns = make_data_dict(texts, wav_paths, labels)            
            self.wav2text = merge(wav2seg, merge_dur)
        self.fns = [(fn, i) for fn in fns for i, _ in enumerate(self.wav2text[fn])]
        self.split = split

    def __getitem__(self, index):
        fn, i = self.fns[index]
        start, end, text, label = self.wav2text[fn][i]
        return text, label, fn

    def __len__(self):
        return len(self.fns)

    def collater(self, batch):
        texts = [text for text, _, _ in batch]
        labels = [label for _, label, _ in batch]
        fns = [fn for _, _, fn in batch]

        inputs = self.tokenizer(texts, return_tensors='pt', max_length=512, padding=True, truncation=True)
        labels = torch.LongTensor(labels)
        
        return {
            'text_input': inputs, 
            'dementia_labels': labels,
            'sizes': torch.ones(len(labels)),
            'ids': fns,
        }
