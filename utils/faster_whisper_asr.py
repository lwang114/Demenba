import argparse
from evaluate import load
from faster_whisper import WhisperModel
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm


def normalize(text):
    text = text.replace(' -', '')
    text = ' '.join([c for c in text.lower() if c.isalpha() or c.isnumeric()])
    return text

def transcribe(wav_path, start, end, model):
    texts = []
    segments_all = []
    
    y, sr = torchaudio.load(wav_path)

    y_trunc = y[:, start:end]
    trunc_wav_path = './trunc.wav'
    torchaudio.save(trunc_wav_path, y_trunc, sr)
        
    segments, _ = model.transcribe(trunc_wav_path)      
    text = [s.text for s in segments]
    text = ' '.join(text)
    text = text.lstrip()
    if not len(text):
        text = '<SIL>'
    return text

parser = argparse.ArgumentParser()
parser.add_argument('--tsv_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/all.tsv')
parser.add_argument('--ref_path', default='')
parser.add_argument('--out_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/all_whisper.wrd')

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
    open(args.out_path, 'w') as f_out:
    lines = f_tsv.read().strip().split('\n')
    root = Path(lines.pop(0))

    preds = []
    for i, l in tqdm(list(enumerate(lines))):
        if i > 2:  # XXX
            break
        wav_fn, start, end = l.strip().split('\t')
        pred = transcribe(str(root / wav_fn), start, end, model)
        print(f'{wav_fn}\t{start}\t{end}\t{pred}', file=f_out, flush=True)
        preds.append(pred)

    if len(refs):
        keep = [i for i, (p, g) in enumerate(zip(preds, refs)) if len(p) and len(g)]
        preds = [preds[i] for i in keep]
        refs = [refs[i] for i in keep]
        print('WER:', 100 * wer.compute(references=refs, predictions=preds))
        print('CER:', 100 * cer.compute(references=refs, predictions=preds))
        with open(args.out_path + '.result', 'w') as f_res:
            print(f'WER: {100 * wer.compute(references=refs, predictions=preds)}', file=f_res)
            print(f'CER: {100 * cer.compute(references=refs, predictions=preds)}', file=f_res) 
