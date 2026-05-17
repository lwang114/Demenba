from collections import defaultdict
import os
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
DOWNSAMPLE_FACTOR=320


class FHSDataset(Dataset):
    def __init__(
            self,
            manifest_dir='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs_split_into_chunks',
            metadata_path='/data/sls/d/fhs/np_dvoice/add_cog_data_(5586)_[11717]_20250131_14_27_9_0179_dvoice_and_npath_add_dr.csv', 
            split='train',
            has_review_only=True,
            merge_dur=30,
            n_class=3,
            num_test_per_class=10,
            num_all_per_class=50,
            add_silence=True):
        self.metadata = pd.read_csv(metadata_path)
        if has_review_only:
            self.metadata = self.metadata[self.metadata['has_dementia_review'] > 0]
        id_dates = self.metadata['id_date'].tolist()
        has_reviews = self.metadata['has_dementia_review'].tolist()
        is_norms = self.metadata['is_norm'].tolist()
        is_mcis = self.metadata['is_mci'].tolist()
        is_dementeds = self.metadata['is_demented'].tolist()
        print(f'Total number of recordings: {len(self.metadata)}', flush=True)

        manifest_dir = Path(manifest_dir)
        self.split = split
        self.merge_dur = merge_dur
        self.add_silence = add_silence

        if n_class == 1:
            self.n_class = 2
        else:
            self.n_class = n_class
        count = torch.zeros(self.n_class)
        self.train_id_dates = []
        self.valid_id_dates = []
        with open(manifest_dir / f'all.tsv', 'r') as f_tsv:
            lines = f_tsv.read().strip().split('\n')
            root = lines.pop(0)
            data_dict = defaultdict(list)
            for l in tqdm(lines):
                wav_path = os.path.join(root, l.strip().split('\t')[0])
                id_date = Path(wav_path).parent.name
                start, end = 0, 0
                if merge_dur > 0:
                    start, end = Path(wav_path).stem.split('_')[-2:]
                    start = int(start)
                    end = int(end)

                if id_date in id_dates:
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
                    data_dict[id_date].append([start, end, [wav_path], label])

                    if count[label] < num_test_per_class:
                        if not id_date in self.valid_id_dates:
                            count[label] += 1
                            self.valid_id_dates.append(id_date)
                    elif count[label] < num_all_per_class: 
                        if not id_date in self.train_id_dates:
                            count[label] += 1
                            self.train_id_dates.append(id_date)
                    elif count.sum() >= self.n_class*num_all_per_class:  # XXX
                        break

        if split == 'train':
            data_dict = {id_date:data_dict[id_date] for id_date in self.train_id_dates}
            if merge_dur > 0:
                data_dict = self.merge(data_dict, merge_dur)
            self.wav_list = [x for id_date in self.train_id_dates for x in data_dict[id_date]]
            print(f'Number of {split} files: {len(self.wav_list)} from {len(self.train_id_dates)} recordings')
        else:
            data_dict = {id_date:data_dict[id_date] for id_date in self.valid_id_dates}
            if merge_dur > 0:
                data_dict = self.merge(data_dict, merge_dur)
            self.wav_list = [x for id_date in self.valid_id_dates for x in data_dict[id_date]]
            print(f'Number of {split} files: {len(self.wav_list)} from {len(self.valid_id_dates)} recordings')

    def __getitem__(self, index):
        _, _, wav_paths, label = self.wav_list[index]
        id_date = Path(wav_paths[0]).parent.name
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
        #if x.shape[-1] > 480000:
        #    x = x[:, :480000]

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
            # print('new_flist[:5]:', new_flist[:5])
            new_data_dict[id_date].extend(new_flist)
            
        print(f'Created {total_merged} merged files from {total} files', flush=True)
        return new_data_dict
