# -*- coding: utf-8 -*-
# Modified from gopt: https://github.com/YuanGongND/gopt

# train and test the models
import argparse
import math
import numpy as np
import sys
import os
import os.path as osp
import pandas as pd
import time
import torch
import torch.nn as nn
from transformers import AutoModel
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from models import *
from dataloaders import * 

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", choices={'train', 'eval', 'eval_gold'}, default='train')
parser.add_argument("--test-split", choices={'train', 'test'}, default='test')
parser.add_argument("--data-dir", type=str, default="../data/fhs", help="directory containing extracted features and labels")
parser.add_argument('--metadata-path', type=str, default='/data/sls/d/fhs/np_dvoice/add_cog_data_(5586)_[11717]_20250131_14_27_9_0179_dvoice_and_npath_add_dr.csv')
parser.add_argument("--lm", type=str, default='bert-base-cased')
parser.add_argument("--classifier", type=str, default='mlp')
parser.add_argument('--download-root', default='/data/sls/scratch/limingw/models/whisper')
parser.add_argument("--layers", type=str, default="23")
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--ce-weight', type=float, default=1.0)
parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--n-epochs", type=int, default=40, help="number of maximum training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument('--topk', type=str, default='2,4,6,8,10,20,40,80,120,160', help='Top-k scores for each recording used for majority voting')
parser.add_argument('--vote-type', type=str, default='hard', choices={'hard', 'soft'})
parser.add_argument('--add-silence', type=int, default=0)
parser.add_argument('--merge-dur', type=float, default=30)
parser.add_argument('--n_class', type=int, default=3)
parser.add_argument('--num-train', type=int, default=50)
parser.add_argument('--text-type', default='raw')

def majority_vote(pred_labels, pred_scores, topk, vote_type='hard', n_class=1):
    if n_class == 1:
        top_indices = sorted(np.arange(len(pred_scores)), key=lambda x:max(pred_scores[x], 1-pred_scores[x]), reverse=True)[:topk]
        vote = np.zeros(2)
        prob = 0.0
        for i in top_indices:
            vote[int(pred_labels[i])] += 1
            prob += pred_scores[i]
    else:
        n_class = pred_scores[0].shape[-1]
        top_indices = sorted(np.arange(len(pred_scores)), key=lambda x:pred_scores[x].max(), reverse=True)[:topk]
        vote = np.zeros(n_class)
        prob = np.zeros(n_class)
        for i in top_indices:
            vote[int(pred_labels[i])] += 1
            prob += pred_scores[i]
    return int(vote.argmax()), prob / len(top_indices)       

def length_to_mask(length, max_len):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    device = length.device
    dtype = length.dtype 
    mask = torch.arange(
        max_len, device=device, dtype=dtype
    ).repeat(len(length), 1) < length.unsqueeze(1)
    return mask

def train(text_model, train_loader, test_loader, args):
    header = [
        'Epoch',
        'Step',
        'Train Loss',
        'Top k',
        'Test Precision',
        'Test Recall',
        'Test Micro F1',
        'Test Macro F1',
        'Test AUC',
        'Best AUC',
    ]
    result = {k:[] for k in header}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))

    best_epoch, best_loss, best_auc = 0, np.inf, 0
    global_step, epoch = 0, 0
    exp_dir = Path(args.exp_dir)
    (exp_dir / 'models').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'preds').mkdir(parents=True, exist_ok=True)

    if not isinstance(text_model, nn.DataParallel):
        text_model = nn.DataParallel(text_model)

    text_model = text_model.to(device)
    # Set up the optimizer
    trainables = [p for p in text_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in text_model.parameters()) / 1e3), flush=True)
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3), flush=True)
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)

    if args.loss == 'bce': 
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'wce':
        if args.n_class == 2:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 3.]).to(device))
        else:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 3., 3.]).to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch), flush=True)
    print("start training...", flush=True)

    begin_time = time.time()
    total_loss = 0
    for epoch in range(args.n_epochs):
        text_model.train()
        for i, batch in enumerate(train_loader):
            text_input = batch['text_input']
            label = batch['dementia_labels']
            sizes = batch['sizes']
            text_input = text_input.to(device)
            label = label.to(device, non_blocking=True)
            sizes = sizes.to(device, non_blocking=True)

            # warmup
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']), flush=True)

            logits = text_model(text_input, sizes)
            if args.loss == 'bce':
                loss = loss_fn(logits.flatten(), label.flatten().float())
            else:
                loss = loss_fn(logits.view(-1, logits.size(-1)), label.flatten())

            loss = args.ce_weight * loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += loss.detach().cpu().item()

            if global_step % 1000 == 0:
                print('start validation', flush=True)
                te_prec_dict, te_rec_dict,\
                te_micro_f1_dict, te_macro_f1_dict,\
                te_auc_dict, te_confusion = validate(text_model, test_loader, args, best_loss)
                if max(te_auc_dict.values()) > best_auc:
                    best_auc = max(te_auc_dict.values())
                    best_epoch = epoch
                    torch.save(text_model.state_dict(), exp_dir / 'models/best_text_model.pth')

                for topk in sorted(te_auc_dict):
                    te_prec = te_prec_dict[topk]
                    te_rec = te_rec_dict[topk]
                    te_micro_f1 = te_micro_f1_dict[topk]
                    te_macro_f1 = te_macro_f1_dict[topk]
                    te_auc = te_auc_dict[topk]
                    print(f'Top {topk}, Precision: {te_prec:.3f}, Recall: {te_rec:.3f}, Micro F1: {te_micro_f1:.3f}, Macro F1: {te_macro_f1:.3f}, AUC: {te_auc:.3f}, Best AUC: {best_auc:.3f}', flush=True)

                    avg_loss = total_loss / global_step
                    result['Epoch'].append(epoch)
                    result['Step'].append(global_step)
                    result['Top k'].append(topk)
                    result['Train Loss'].append(avg_loss)
                    result['Test Precision'].append(te_prec)
                    result['Test Recall'].append(te_rec)
                    result['Test Micro F1'].append(te_micro_f1)
                    result['Test Macro F1'].append(te_macro_f1)
                    result['Test AUC'].append(te_auc)
                    result['Best AUC'].append(best_auc)
                df = pd.DataFrame(result)
                df.to_csv(str(exp_dir / 'result.csv'))
                print('-------------------validation finished-------------------', flush=True)

        if global_step > warm_up_step:
            scheduler.step()

        print('Epoch-{0} lr: {1}, takes {2}s'.format(epoch, optimizer.param_groups[0]['lr'], time.time()-begin_time), flush=True)
    print(f'Total training time: {time.time()-begin_time}s')


def validate(text_model, val_loader, args, best_loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(text_model, nn.DataParallel):
        text_model = nn.DataParallel(text_model)
    text_model = text_model.to(device)
    text_model.eval()

    confusion = np.zeros([2, 2])
    pred_scores_dict = defaultdict(list)
    pred_labels_dict = defaultdict(list)
    gold_labels_dict = {}
    exp_dir = Path(args.exp_dir)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            text_input = batch['text_input']
            sizes = batch['sizes']
            dementia_labels = batch['dementia_labels']
            id_dates = batch['ids']
            text_input = text_input.to(device)
            dementia_labels = dementia_labels.to(device)
            sizes = sizes.to(device)

            logits = text_model(text_input, sizes)
            if args.loss == 'bce':
                pred_labels = (logits > 0).long()
            else:
                pred_labels = logits.argmax(-1)

            for logit, label, pred_label, id_date in zip(logits, dementia_labels, pred_labels, id_dates):
                if args.loss != 'bce':
                    pred_scores_dict[id_date].append(logit.softmax(-1).detach().cpu().numpy())
                else:
                    pred_scores_dict[id_date].append(logit.sigmoid().detach().cpu().item())

                pred_labels_dict[id_date].append(pred_label.detach().cpu().item())
                gold_labels_dict[id_date] = label.detach().cpu().item()

        prec_dict, rec_dict, micro_f1_dict, macro_f1_dict, auc_dict = {}, {}, {}, {}, {}
        for topk in map(int, args.topk.split(',')):
            all_pred_scores = []
            all_pred_labels = []
            all_gold_labels = []
            for id_date, gold_label in sorted(gold_labels_dict.items()):
                pred_label, pred_score = majority_vote(pred_labels_dict[id_date], pred_scores_dict[id_date], topk=topk, n_class=args.n_class)
                all_pred_labels.append(pred_label)
                all_pred_scores.append(pred_score)
                all_gold_labels.append(gold_label)
            
            pred_labels = np.asarray(all_pred_labels)
            pred_scores = np.asarray(all_pred_scores)
            gold_labels = np.asarray(all_gold_labels)
            if args.loss != 'bce':
                gold_labels_onehot = np.zeros((gold_labels.shape[0], args.n_class))
                gold_labels_onehot[gold_labels] = 1
                auc = np.asarray([
                    roc_auc_score(gold_labels_onehot[:, k], pred_scores[:, k])
                    for k in range(args.n_class)
                ]).mean()
            else:
                auc = roc_auc_score(gold_labels, pred_scores)

            _, _, micro_f1, _ = precision_recall_fscore_support(
                gold_labels, pred_labels, average='micro',
            )
            precision, recall, macro_f1, _ = precision_recall_fscore_support(
                gold_labels, pred_labels, average='macro',
            )

            prec_dict[topk] = precision 
            rec_dict[topk] = recall
            micro_f1_dict[topk] = micro_f1
            macro_f1_dict[topk] = macro_f1
            auc_dict[topk] = auc

            confusion = confusion_matrix(gold_labels, pred_labels)
            np.save(exp_dir / 'preds' / f'top{topk}_confusion.npy', confusion)
            np.save(exp_dir / 'preds' / f'top{topk}_{val_loader.dataset.split}_gold_label.npy', gold_labels)
            np.save(exp_dir / 'preds' / f'top{topk}_{val_loader.dataset.split}_pred_label.npy', pred_labels)
            if len(pred_scores):
                np.save(exp_dir / 'preds' / f'top{topk}_{val_loader.dataset.split}_pred_scores.npy', pred_scores)

    return prec_dict, rec_dict, micro_f1_dict, macro_f1_dict, auc_dict, confusion

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args.exp_dir)
layers = list(map(int, args.layers.split(',')))

lm = args.lm
feat_dim = {
    'bert-base-cased': 768,
}
input_dim = feat_dim[lm]

# Text model
bert_model = AutoModel.from_pretrained(args.lm)
classifier = BERTMLPClassifier(bert_model, 768, 2, input_dim)
print(classifier)

if args.mode == 'eval_gold':
    te_dataset = FHSGoldTextDataset(
        manifest_dir=args.data_dir,
        split=args.test_split,
        merge_dur=args.merge_dur,
        n_class=args.n_class,
        feat_type=args.lm,
        text_type=args.text_type,
    )
else:
    te_dataset = TextDataset(
        manifest_dir=args.data_dir,
        split='test',
        num_all_per_class=args.num_train,
        merge_dur=args.merge_dur,
        n_class=args.n_class,
        feat_type=args.lm,
    )
    #te_dataset = FHSGoldDataset(
    #    args.data_dir, split=args.test_split, merge_dur=args.merge_dur,
    #)

# te_dataloader = DataLoader(te_dataset, collate_fn=te_dataset.collater, batch_size=args.batch_size, shuffle=False)
te_dataloader = DataLoader(
    te_dataset,
    batch_size=args.batch_size, 
    collate_fn=te_dataset.collater,
    shuffle=False,
)

if args.mode == 'train':
    tr_dataset = TextDataset(
        manifest_dir=args.data_dir,
        split='train',
        num_all_per_class=args.num_train,
        merge_dur=args.merge_dur,
        n_class=args.n_class,
        feat_type=args.lm,
    )
#    tr_dataloader = DataLoader(tr_dataset, collate_fn=tr_dataset.collater, batch_size=args.batch_size, shuffle=True)
    tr_dataloader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        collate_fn=tr_dataset.collater,
        shuffle=True,
    )
    train(classifier, tr_dataloader, te_dataloader, args)
else:
    exp_dir = Path(args.exp_dir) 
    classifier = nn.DataParallel(classifier)
    classifier.load_state_dict(torch.load(exp_dir / 'models/best_text_model.pth'))

    te_prec_dict, te_rec_dict,\
    te_micro_f1_dict, te_macro_f1_dict,\
    te_auc_dict, te_confusion = validate(classifier, te_dataloader, args, -1)

    for topk in te_prec_dict:
        print(f'Top {topk}, Precision: {te_prec_dict[topk]:.3f}, Recall: {te_rec_dict[topk]:.3f}, Micro F1: {te_micro_f1_dict[topk]:.3f}, Macro F1: {te_macro_f1_dict[topk]:.3f}, AUC: {te_auc_dict[topk]:.3f}', flush=True)
    print('-------------------validation finished-------------------', flush=True)
