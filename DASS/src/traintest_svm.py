# -*- coding: utf-8 -*-

# train and test the models
import argparse
from collections import defaultdict, Counter
import numpy as np
import sys
import os
import os.path as osp
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix 
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.compute_auc import compute_auc

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
ONSET_SITE = ['not_documented', 'Arm', 'Breathing', 'Foot', 'Hand', 'Leg', 'Swallowing', 'Tongue'] 

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data-dir", type=str, default="../data/als/with_score/whisper_large-v2/feat_mean+concat", help="directory containing extracted features and labels")
parser.add_argument("--layers", type=str, default="31")
parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
parser.add_argument("--model", type=str, default='svc')
parser.add_argument("--label-name", type=str, default='score')

def gen_result_header():
    header = [
        'epoch', 
        'test_precision', 
        'test_recall', 
        'test_micro_f1', 
        'test_macro_f1',
        'test_auc',
    ]
    return header

def load_feats(paths):
    X_list = []
    spk2idx = defaultdict(list)
    for p in paths:
        X = np.load(p+'.npy')
        with open(p+'.lengths', 'r') as f_size,\
            open(p+'.tsv', 'r') as f_tsv:
            sizes = [int(s) for l in f_size.read().strip().split('\n') for s in l.split()]

            lines = f_tsv.read().strip().split('\n')
            _ = lines.pop(0)
            if not len(spk2idx):
                for i, l in enumerate(lines):
                    spk = os.path.basename(l.strip().split()[0]).split('_')[0]
                    spk2idx[spk].append(i)

        X_seg = []
        offset = 0
        for size in sizes:
            X_seg.append(X[offset:offset+size].mean(0))
            offset += size
        X_seg = np.stack(X_seg)
        X_list.append(X_seg)
    X = np.concatenate(X_list, axis=-1)
    return X, spk2idx

args = parser.parse_args()
np.random.seed(args.seed)
print(f'Target label: {args.label_name}', flush=True)
if len(args.layers):
    data_dirs = [osp.join(args.data_dir, f'layer{l}') for l in args.layers.split(',')]
else:
    data_dirs = [args.data_dir]
n_weights = len(data_dirs)

tr_paths = [osp.join(data_dir, 'train') for data_dir in data_dirs]
te_paths = [osp.join(data_dir, 'test') for data_dir in data_dirs]

X_tr, _ = load_feats(tr_paths) 
X_te, spk2idx = load_feats(te_paths) 

with open(tr_paths[0] + '.' + args.label_name, 'r') as f:
    lines = f.read().strip().split('\n')
    Y_tr = [y for l in lines for y in list(map(int, l.strip().split()))]
    Y_tr = np.asarray(Y_tr)

with open(te_paths[0] + '.' + args.label_name, 'r') as f:
    lines = f.read().strip().split('\n')
    Y_te = [y for l in lines for y in list(map(int, l.strip().split()))]
    Y_te = np.asarray(Y_te)

print(X_tr.shape, Y_tr.shape, X_te.shape, Y_te.shape, flush=True)
assert (X_tr.shape[0] == Y_tr.shape[0]) and (X_te.shape[0] == Y_te.shape[0])

if args.model == 'svc':
    svm = SVC(kernel='linear', probability=True)
elif args.model == 'svr':
    svm = SVR(kernel='linear')
    Y_tr = Y_tr.astype(float)
elif args.model == 'linear_svc':
    svm = LinearSVC()
else:
    svm = SGDClassifier()

Y_tr_pred = svm.fit(X_tr, Y_tr).predict(X_tr)
print((Y_tr_pred == Y_tr).mean(), flush=True)  # XXX

Y_te_pred = svm.predict(X_te)

proba = None
if args.model == 'svc':
    proba = svm.predict_proba(X_te)
_, _, micro_f1, _ = precision_recall_fscore_support(Y_te, Y_te_pred, average='micro') 
precision, recall, macro_f1, _ = precision_recall_fscore_support(Y_te, Y_te_pred, average='macro')

auc = 0
if proba is not None:
    auc = compute_auc(Y_te, proba)
confusion = confusion_matrix(Y_te, Y_te_pred)

exp_dir = Path(args.exp_dir)
exp_dir.mkdir(parents=True, exist_ok=True)
result = np.zeros([1, 6])
result[0, 1:6] = [precision, recall, micro_f1, macro_f1, auc]
header = ','.join(gen_result_header())
if proba is not None:
    np.save(exp_dir / 'proba.npy', proba)
np.save(exp_dir / 'pred.npy', Y_te_pred)
np.save(exp_dir / 'gold.npy', Y_te)
np.savetxt(exp_dir / 'confusion.csv', confusion, fmt='%d', delimiter=',')
np.savetxt(exp_dir / 'result.csv', result, fmt='%.4f', delimiter=',', header=header, comments='')
print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, Micro F1: {micro_f1:.3f}, Macro F1: {macro_f1:.3f}, AUC: {auc:.3f}')
