import argparse
from evaluate import load
from pathlib import Path
import torch
import torchaudio
from faster_whisper import WhisperModel
from tqdm import tqdm


def normalize(text):
    text = text.replace(' -', '')
    text = ' '.join([c for c in text.lower() if c.isalpha() or c.isnumeric()])
    return text

def transcribe(wav_path, model):
    segments, _ = model.transcribe(wav_path)
    text = [s.text for s in segments]
    text = ' '.join(text)
    text = text.lstrip()
    if not len(text):
        text = '<SIL>'
    return text

parser = argparse.ArgumentParser()
parser.add_argument('--tsv_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/all.tsv')
parser.add_argument('--ref_path', default='')
parser.add_argument('--out_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/all_whisper')
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=200)

args = parser.parse_args()

wer = load('wer')
cer = load('cer')

model_name = 'large-v2'
#model_name = 'distil-large-v3'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WhisperModel(model_name, device=device, compute_type='float16')

refs = []
if args.ref_path:
    with open(args.ref_path, 'r') as f:
        lines = f.read().strip().split('\n')
        refs = [normalize(l) for l in lines]

with open(args.tsv_path, 'r') as f_tsv,\
    open(f'{args.out_path}_{args.start_idx}_{args.end_idx}.wrd', 'w') as f_out:
    lines = f_tsv.read().strip().split('\n')
    root = Path(lines.pop(0))

    wav_fns = []
    preds = []
    idx = 0
    for i, l in tqdm(list(enumerate(lines))):
        wav_fn, start, end = l.strip().split('\t')
        if (not wav_fn in wav_fns) and (idx >= args.start_idx) and (idx < args.end_idx):
            pred = transcribe(str(root / wav_fn), model)
            print(f'{wav_fn}\t{pred}', file=f_out, flush=True)
            preds.append(pred)
            wav_fns.append(wav_fn)
            idx += 1

    if len(refs):
        keep = [i for i, (p, g) in enumerate(zip(preds, refs)) if len(p) and len(g)]
        preds = [preds[i] for i in keep]
        refs = [refs[i] for i in keep]
        print('WER:', 100 * wer.compute(references=refs, predictions=preds))
        print('CER:', 100 * cer.compute(references=refs, predictions=preds))
        with open(args.out_path + '.result', 'w') as f_res:
            print(f'WER: {100 * wer.compute(references=refs, predictions=preds)}', file=f_res)
            print(f'CER: {100 * cer.compute(references=refs, predictions=preds)}', file=f_res) 
