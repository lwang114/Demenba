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
from tqdm import tqdm
DOWNSAMPLE_FACTOR=320


def make_data_dict(audio_dir, wav_paths, labels, speakers):
    '''
    Create a dictionary that maps wav ids to a list of tuples [start, end, [wav_path], label]
    '''
    data_dict = defaultdict(list)
    for p, y in zip(wav_paths, labels):
        parts = Path(p).stem.split('_')
        wav_id = '_'.join(parts[:4])

        if parts[1] == parts[4]:
            spk = 'Participant'
        else:
            spk = 'Interviewer'
        
        if spk in speakers:
            start, end = parts[-2:]
            start = int(start)
            end = int(end)
            data_dict[wav_id].append([start, end, [p], y])
    return data_dict


class FHSGoldDataset(Dataset):
    def __init__(
            self,
            audio_dir='/data/sls/d/corpora/original/FHS/FHS_2022_Gold92/all_audios_16kHz/', 
            folder='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs_gold92',
            split='all',
            audio_conf={},
            merge_dur=360,
            n_class=2,
            add_silence=True,
            speakers=['Interviewer', 'Participant'],
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
            if n_class == 2:
                labels = list(map(lambda x: int(x > 0), labels))
            self.label_num = len(set(labels))
            self.pos_num = sum(labels)
            self.total_num = len(labels)
            assert(len(labels) == len(wav_paths), "number of wavs and number of labels should match")
            print(f'Number of classes: {self.label_num}')
      
        self.n_class = n_class
        self.audio_conf = audio_conf 
        # dataset spectrogram mean and std, used to normalize the input
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')

        data_dict = make_data_dict(audio_dir, wav_paths, labels, speakers)
        data_dict = self.merge(data_dict, merge_dur)
        self.wav_list = [x+[id_date] for id_date in data_dict for x in data_dict[id_date]]

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        start, end, wav_paths, label, id_date = self.wav_list[index]
        xs = []
        for wav_path in wav_paths:
            x, sr = torchaudio.load(wav_path)
            if sr != 16000:
                x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=16000)
            xs.append(x)
        x = torch.cat(xs, dim=-1)
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

        # fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
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
