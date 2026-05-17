import argparse
from collections import defaultdict
import json
import re
from tqdm import tqdm


def create_prompt(text, label_set, prompt_type='simple'):
    prompt = (
        "You are a helpful assistant that classifies if a participant in an interview has dementia.\n\n"
        f"Possible labels: {', '.join(label_set)}.\n"
        f"Interview transcript: \"{text}\"\n"
        "Label:"
    )    
    return prompt

def merge_speaker_text(sents):
    prompt = ''
    prev_spk = ''
    for sent in sents:
        spk = sent.split(': ')[0]
        sent = ': '.join(sent.split(': ')[1:])
        if spk != prev_spk:
            if prev_spk:
                prompt += f'\n{spk}: {sent}'
            else:
                prompt += f'{spk}: {sent}'
        else:
            prompt += f' {sent}'
        prev_spk = spk
    return prompt

def replace_non_english(text):
    # Match any character that is not English alphabet, digit, common punctuation or space
    return re.sub(r'[^\x00-\x7F]+', '<SPOKEN_NOISE>', text)

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

parser = argparse.ArgumentParser()
parser.add_argument('--wrd-path', default='../data/train_800_diarized_whisper.wrd')
parser.add_argument('--tsv-path', default='../data/train_800.tsv')
parser.add_argument('--lbl-path', default='../data/train_800.lbl')
parser.add_argument('--out-path', default='../data/train_800.json')
parser.add_argument('--label-set', default='normal,dementia')
parser.add_argument('--merge-dur', type=int, default=360)
args = parser.parse_args()

label_set = args.label_set.split(',')

with open(args.wrd_path, 'r') as f:
    lines = f.read().strip().split('\n')
    wav2seg = defaultdict(list)
    for l in lines:
        fn, start, end = l.strip().split('\t')[:3]
        text = ' '.join(l.strip().split('\t')[3:])
        wav2seg[fn].append([int(start), int(end), text])
    wav2text = merge(wav2seg, args.merge_dur)

with open(args.tsv_path, 'r') as f,\
    open(args.lbl_path, 'r') as f_lbl:
    lines = f.read().strip().split('\n')
    root = lines.pop(0)   
    labels = f_lbl.read().strip().split('\n')

    filenames = []
    wav2lab = {}
    for l, y in zip(lines, labels):
        fn = l.strip().split('\t')[0]
        if not fn in filenames:
            filenames.append(fn)
            wav2lab[fn] = int(y)

data_dict = []
for fn in filenames:
    if fn in wav2text:
        texts = wav2text[fn]  #merge_text(wav2text[fn])
        for start, end, text in texts:
            prompt = create_prompt(text, label_set=label_set)
            prompt = replace_non_english(prompt)
            answer = label_set[wav2lab[fn]]
            new_case = {
                'name': f'{fn}_{start}_{end}',
                'messages': [
                    {'content': prompt, 'role': 'user'},
                    {'content': answer, 'role': 'assistant'},
                ],
            }
            data_dict.append(new_case)

with open(args.out_path, 'w') as f:
    json.dump(data_dict, f, indent=2, sort_keys=True)
