import argparse
from collections import defaultdict
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diarize-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100_diarized.tsv')
    parser.add_argument('--asr-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100_whisper.wrd')
    parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100_diarized_whisper.wrd')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.diarize_path, 'r') as f_dia,\
        open(args.asr_path, 'r') as f_asr,\
        open(args.out_path, 'w') as f_out:
        dia_lines = f_dia.read().strip().split('\n')
        asr_lines = f_asr.read().strip().split('\n')
        root = dia_lines.pop(0)

        fn2diar = defaultdict(list)
        for dia_line in tqdm(dia_lines):
            fn, start, end, spk = dia_line.split('\t')
            fn2diar[fn].append([start, end, spk])

        prev_fn = ''
        spk_count = 0
        for asr_line in tqdm(asr_lines):
            fn = asr_line.split('\t')[0]
            text = '\t'.join(asr_line.split('\t')[1:])
            if fn != prev_fn:
                spk_count = 0
                prev_fn = fn
            if not fn in fn2diar:
                print(f'Warning: No diarization found for {fn}')
            else:
                start, end, spk = fn2diar[fn][spk_count]
                spk_count += 1
                print(f'{fn}\t{start}\t{end}\t{spk}: {text}', file=f_out)

if __name__ == '__main__':
    main()
