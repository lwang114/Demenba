import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='/data/sls/d/fhs/np_dvoice/clean_dvrs/2025-01-21')
parser.add_argument('--out_dir', default='../data/fhs/wavs_16khz')
args = parser.parse_args()

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)
for spk_dir in in_dir.iterdir():
    spk = spk_dir.name
    for date_dir in spk_dir.iterdir():
        date = date_dir.name
        for wav_file in date_dir.iterdir():
            wav_id = wav_file.name.split('.')[0]
            out_path = out_dir / spk / date / f'{wav_id}.wav'
            if out_path.exists():
                continue
            print(out_path)
            command = ['ffmpeg', '-i', str(wav_file), '-ar', '16000', '-ac', '1', '-y', str(out_path)]
            subprocess.run(command)
