import argparse
from pathlib import Path
from pyannote.audio import Pipeline
import torch
from tqdm import tqdm
import os
HF_TOKEN = os.environ.get("HF_TOKEN")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100.tsv')
    parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100_diarized.tsv')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    sr = 16000

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN)

    # send pipeline to GPU (when available)
    pipeline.to(torch.device("cuda"))

    fns = []
    with open(args.tsv_path, 'r') as f,\
        open(args.out_path, 'w') as f_out:
        lines = f.read().strip().split('\n')
        root = lines.pop(0)
        print(root, file=f_out)
        for l in tqdm(lines):
            fn = l.strip().split('\t')[0]
            if not fn in fns:
                fns.append(fn)
            else:
                continue
            wav_path = Path(root) / fn

            # apply pretrained pipeline
            diarization = pipeline(str(wav_path))

            interviewer_id = None 
            spks = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                dur = turn.end - turn.start
                if dur > 0.2:
                    if not speaker in spks:
                        spks.append(speaker)

                    if interviewer_id is None:
                        interviewer_id = speaker
                        print(f'{fn} interviewer: {speaker}')

                    start = int(turn.start * sr)
                    end = int(turn.end * sr)
                    # print the result
                    if speaker == interviewer_id:
                        print(f'{fn}\t{start}\t{end}\tInterviewer', file=f_out)
                    else:
                        print(f'{fn}\t{start}\t{end}\tParticipant', file=f_out)
            print(f'{fn} number of speakers: {len(spks)}')

if __name__ == '__main__':
    main()
