# -*- coding: utf-8 -*-
import argparse
import whisper
from npy_append_array import NpyAppendArray
import os
import os.path as osp
from pathlib import Path
from shutil import copyfile

import torch
import tqdm
UNK = '<unk>'


def get_parser():
    parser = argparse.ArgumentParser()  
    parser.add_argument('data', help='location of .npy files')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--model-name', help='which whisper model to use')
    parser.add_argument('--download-root', help='where to download the whisper model', default='/data/sls/scratch/limingw/workplace/models')
    parser.add_argument('--layers', help='the layers to extract the features', default='23')
    parser.add_argument('--pooling', default='mean')
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=-1)
    return parser


class WhisperReader(object):
    def __init__(self, model_name, download_root, layers):
        self.model = whisper.load_model(model_name, download_root=download_root).cuda()
        self.layers = layers

    def read_audio(self, fnames, sr=16e3):
        audio_list = []
        for fname in fnames:
            audio = whisper.load_audio(fname)
            audio_list.append(audio)
        audio = torch.cat(audio_list)
        audio_len = len(audio) // 320
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        return mel, audio_len

    def normalize(self, w):
        w = w.strip()
        w = ''.join([c for c in w if c.isalpha()])
        return w

    def get_feats(self, locs):
        '''Return encoder features, word-level labels and segmentations'''
        mel, audio_len = self.read_audio(locs)

        layer_results = {}
        def get_layer_results(name):
            def hook(model, input, output):
                layer_results[name] = output.detach()
            return hook

        for l in self.layers:
            self.model.encoder.blocks[l].register_forward_hook(get_layer_results(f'block_{l}'))
    
        with torch.no_grad():
            _ = self.model.encoder(mel.unsqueeze(0))
            feats = []
            for l in self.layers:
                feat = layer_results[f'block_{l}'].contiguous().squeeze(0).cpu()
                feats.append(feat[:audio_len])

        return feats

def get_iterator(args):
    with open(osp.join(args.data, args.split) + '.tsv', 'r') as fp,\
        open(osp.join(args.data, args.split) + '_cat.tsv', 'w') as fc:
        lines = fp.read().split('\n')
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split('\t')[0]) for line in lines if len(line) > 0]
        durs = [int(line.split('\t')[1]) / 16e3 for line in lines if len(line) > 0] 
        # Concatenate the audio files to around 30s
        files_to_cat = []
        file_list = []
        prev_id = Path(files[0]).parent.name
        cat_dur = 0
        for fn, dur in zip(files, durs):
            cur_id = Path(fn).parent.name
            if (cur_id != prev_id) or (dur > 30):
                files_to_cat.append(file_list)
                file_list = [str(fn)]
                cat_dur = dur
            else:
                file_list.append(fn)
                cat_dur += dur
            prev_id = cur_id
        files_to_cat.append(file_list)

        layers = list(map(int, args.layers.split(',')))
   
        num = len(files_to_cat)

        if args.end_idx == -1:
            args.end_idx = num
            files_to_cat = files_to_cat[args.start_idx:args.end_idx]

        reader = WhisperReader(args.model_name, args.download_root, layers)

        def iterate():
            for file_list in files_to_cat:
                print('_'.join(file_list), file=fc)
                embed = reader.get_feats(file_list)
                yield embed

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()
    layers = args.layers.split(',')

    print(args.save_dir)
    for l in layers:
        os.makedirs(osp.join(args.save_dir, f'layer{l}'), exist_ok=True)

    def create_files(dest):
        copyfile(osp.join(args.data, args.split) + ".tsv", dest + ".tsv")
        npaa = NpyAppendArray(dest + '.npy', delete_if_exists=True)
        return npaa

    save_paths = [osp.join(args.save_dir, f'layer{l}', args.split) for l in layers]
    npaas = [create_files(save_path) for save_path in save_paths]
    l_fs = [open(save_path + ".lengths", "w") for save_path in save_paths]

    generator, num = get_iterator(args)
    iterator = generator()

    for feats_list in tqdm.tqdm(iterator, total=num): 
        for l, feats, npaa, l_f in zip(layers, feats_list, npaas, l_fs):
            if args.pooling == 'mean':
                feats = feats.mean(0, keepdim=True)
            print(len(feats), file=l_f)

            if len(feats) > 0:
                npaa.append(feats.numpy())

    for l_f in l_fs:
        l_f.close()

if __name__ == '__main__':
    main()
