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


def merge(data_dict, merge_dur):
    new_data_dict = defaultdict(list)
    total = 0
    total_merged = 0
    for id_date in tqdm(list(sorted(data_dict))):
        flist = sorted(data_dict[id_date], key=lambda x:x[0])
        start, end, text = flist[0]
        new_fn = [start, end, text]
        new_flist = []
        for start, end, text in flist[1:]:
            total += 1
            if end - new_fn[0] > merge_dur * 16e3:
                total_merged += 1
                new_flist.append(new_fn)
                new_fn = [start, end, text]
            else:
                new_fn[1] = end
                new_fn[2] += f'\n{text}'

        new_flist.append(new_fn)
        new_data_dict[id_date].extend(new_flist)
        
    print(f'Created {total_merged} merged files from {total} files', flush=True)
    return new_data_dict


class TextDataset(Dataset):
    def __init__(
            self,
            manifest_dir='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs',
            split='train',
            has_review_only=True,
            merge_dur=30,
            n_class=3,
            num_test_per_class=10,
            num_all_per_class=50,
            add_silence=False,
            feat_type='bert-base-cased',
        ):
        manifest_dir = Path(manifest_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(feat_type, use_fast=True)

        fns = []
        if n_class == 1:
            n_class = 2
        with open(manifest_dir / f'{split}_{num_all_per_class*n_class}.tsv', 'r') as f_tsv,\
            open(manifest_dir / f'{split}_{num_all_per_class*n_class}.lbl', 'r') as f_lbl:
            labels = f_lbl.read().strip().split('\n')
            lines = f_tsv.read().strip().split('\n')
            root = lines.pop(0)
            wav2seg = defaultdict(list)
            self.wav2lab = {}
            for l, y in tqdm(list(zip(lines, labels))):
                fn = l.strip().split('\t')[0]
                if not fn in fns:
                    fns.append(fn)
                    self.wav2lab[fn] = int(y)
        
        with open(manifest_dir / f'{split}_{num_all_per_class*n_class}_diarized_whisper.wrd', 'r') as f:
            texts = f.read().strip().split('\n')
            for l in tqdm(texts):
                fn, start, end = l.strip().split('\t')[:3]
                text = ''.join(l.strip().split('\t')[3:])
                wav2seg[fn].append([int(start), int(end), text])
            self.wav2text = merge(wav2seg, merge_dur)
        self.fns = [(fn, i) for fn in fns for i, _ in enumerate(self.wav2text[fn])]
        self.split = split

    def __getitem__(self, index):
        fn, i = self.fns[index]
        start, end, text = self.wav2text[fn][i]
        label = self.wav2lab[fn]
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
