import argparse
import collections
import contextlib
import pandas as pd
from pathlib import Path

import sys
from tqdm import tqdm
import wave

import webrtcvad


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/data/sls/d/fhs/np_dvoice/add_cog_data_(5586)_[11717]_20250131_14_27_9_0179_dvoice_and_npath_add_dr.csv')
    parser.add_argument('--wav_root', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/wavs_16khz')
    parser.add_argument('--target_dir', default='../data/fhs_split_into_chunks')
    parser.add_argument('--agressiveness', type=int, default=1)
    return parser

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        print(path, num_channels)
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        print(sample_rate) # XXX
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def main():
    parser = get_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    tgt_dir = Path(args.target_dir)
    wav_root = Path(args.wav_root)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    total_dur = 0
    bad_files = []
    for fn in tqdm(df['audio_filepath']):
        print('fn:', fn)
        wav_id = Path(fn).stem
        id_date = Path(fn).parent.name
        wav_path = wav_root / fn
        if not wav_path.exists():
            fn = fn.replace('clean_dvrs/2025-01-21/', '').split('.')[0]+'.wav'
            wav_path = wav_root / fn
        
        try:
            audio, sample_rate = read_wave(str(wav_path))
        except:
            wav_id = wav_id.lower().replace('-', '_')
            wav_path = wav_path.parent / f'{wav_id}.wav'
            try:
                audio, sample_rate = read_wave(str(wav_path))
            except:
                bad_files.append(str(wav_path))
                print(f'Warning: {wav_path} not found, skip')
        
        with open(tgt_dir / 'bad_files.txt', 'w') as f_bad:
            print('\n'.join(bad_files), file=f_bad)

if __name__ == '__main__':
    main()
