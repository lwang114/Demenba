import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/train_800_whisper_part')
parser.add_argument('--n_parts', type=int, default=20)
parser.add_argument('--out_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/train_800_whisper.wrd')

args = parser.parse_args()
total = 0
with open(args.out_path, 'w') as f_out:
    for i in range(args.n_parts+1):
        in_path = f'{args.in_path}_{i}.wrd'
        with open(in_path, 'r') as f:
            lines = f.read().strip().split('\n')
            total += len(lines)
            print(in_path, len(lines))
            print('\n'.join(lines), file=f_out)

print(f'Total number of segments: {total}')
