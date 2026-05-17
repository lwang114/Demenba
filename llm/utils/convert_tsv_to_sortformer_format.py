import argparse
import json
import torchaudio
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--tsv-path', default='../../../data/test_100.lbl')
parser.add_argument('--out-path', default='./data/test_100.jsonl')
parser.add_argument('--segment-dur', type=int, default=600)
parser.add_argument('--overlap-dur', type=int, default=120)

args = parser.parse_args()
wav_fns = []
skip_dur = args.segment_dur - args.overlap_dur

with open(args.tsv_path, 'r') as f,\
    open(args.out_path, 'w') as f_out:
    lines = f.read().strip().split('\n')
    root = Path(lines.pop(0))
    for l in lines:
        wav_fn = l.strip().split('\t')[0]
        if not wav_fn in wav_fns:
            wav_fns.append(wav_fn)
            wav_path = root / wav_fn
            x, sr = torchaudio.load(str(wav_path))
            audio_len = x.shape[-1]
            total_dur = audio_len / sr
            offset = 0
            while offset < total_dur:
                dur = min(args.segment_dur, total_dur - offset) 
                data = {
                    'audio_filepath': str(wav_path),
                    'offset': offset,
                    'dur': dur,
                }
                print(json.dumps(data), file=f_out)
                offset += skip_dur
