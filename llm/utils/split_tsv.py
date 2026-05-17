import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--tsv-path', default='../../../data/fhs/train_800.tsv')
parser.add_argument('--n-parts', type=int, default=10)
parser.add_argument('--out-prefix', default='../../../data/fhs/train_800_part')
args = parser.parse_args()
Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

with open(args.tsv_path, 'r') as f:
    lines = f.read().strip().split('\n')
    root = lines.pop(0)
    part_size = len(lines) // args.n_parts 
    for i in range(args.n_parts):
        with open(f'{args.out_prefix}_{i:02d}.tsv', 'w') as f_out:
            print(root, file=f_out)
            print('\n'.join(lines[i*part_size:(i+1)*part_size]), file=f_out)

    if len(lines) > (i+1)*part_size:
        with open(f'{args.out_prefix}_{i+1:02d}.tsv', 'w') as f_out:
            print(root, file=f_out)
            print('\n'.join(lines[(i+1)*part_size:]), file=f_out)
