# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong, Saurabhchand Bhati
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pandas as pd
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader_fhs_10khr, dataloader_fhs_gold
import models
import numpy as np
from traintest import train, validate

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--manifest_dir", required=True)
parser.add_argument("--metadata_path", required=True)
parser.add_argument("--n_class", type=int, default=2, help="number of classes")
#parser.add_argument("--cross_validation_split", type=float, default=0.9, help="train/val split for cross validation")
#parser.add_argument("--split_seed", type=int, default=0, help="random seed for cross validation split")
parser.add_argument("--seed", type=int, default=42, help="random seed for cross validation split")
parser.add_argument("--path", type=str, default='/data/sls/scratch/amagaro/project/speaker_anonymization/fhs_gold92_long/', help="parent data dir")
parser.add_argument("--folder", type=int, default=1, help="data folder used (1-5)")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--model_size", type=str, default='tiny224', help="the model size used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")
parser.add_argument('--merge_dur', type=float, default=30)
parser.add_argument('--test_merge_dur', default='')
parser.add_argument('--num_all_per_class', type=int, default=50)
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
#parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument('-w', '--num-workers', default=0, type=int, metavar='NW', help='# of workers for dataloading (default: 0)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')
parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=3072, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP", "f1"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE", "WCE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")
parser.add_argument('--pos_emb_type', type=str, default='learned',choices=['learned','none','sine'], help="the type of positional embedding used in the model")
parser.add_argument('--knowledge_distillation', type=ast.literal_eval, default='False', help="use knowledge distillation for the model")
parser.add_argument('--dist_loss_type',type=str,default='kldiv',choices=['l2c','kldiv','cosine','l2','bce'],help="the type of distillation loss used in the model")
parser.add_argument('--dist_loss_weight',type=float,default=0.5,help="the weight of distillation loss in the total loss")
parser.add_argument('--dist_temp',type=float,default=1.0,help="the temperature of distillation loss")
parser.add_argument('--dist_teacher_func',type=str,default='sigmoid',choices=['sigmoid','softmax'],help="the function applied to the teacher model output")
parser.add_argument('--kd_teach_type',type=str,default='ast',choices=['ast','DASS','self'],help="the type of teacher model used for knowledge distillation")
parser.add_argument('--kd_teach_dir',type=str,default='',help="the directory of the teacher model")
parser.add_argument('--kd_EMA_momentum',type=float,default=0.999,help="the momentum of the EMA teacher model")
parser.add_argument('--clf_input_emb',type=str,default='avg',choices=['first','last','mid','max','avg','cls','sum'],help="use input embedding for the classifier")
parser.add_argument('--ssm_ratio',type=float,default=1,help="ssm ratio")
parser.add_argument('--ssm_d_state',type=int,default=1,help="ssm d state")
parser.add_argument('--es_patience',type=int,default=5,help="early stopping patience, -1 means no early stopping")
parser.add_argument('--add_silence',type=ast.literal_eval,default='False')
parser.add_argument('--diarize',type=ast.literal_eval,default='True')
parser.add_argument('--vote',type=ast.literal_eval,default='True')
parser.add_argument('--vote_type',type=str,default='soft')
parser.add_argument('--force_limit_batches_per_epoch',type=float,default=-1,help="effective num batches in epoch = force_limit_batches_per_epoch * len(train_loader), negative value means no limit, only used for full dataset")
parser.add_argument('--mode',type=str,default='train')
parser.add_argument('--speakers',type=str,default='Participant,Interviewer')

args = parser.parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# transformer based model

print('now train a audio spectrogram transformer model')

# 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
# dataset spectrogram mean and std, used to normalize the input
# norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
# target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
# # if add noise for data augmentation, only use for speech commands
# noise = {'audioset': False, 'esc50': False, 'speechcommands':True}

audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                'noise':args.noise}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}
if not len(args.test_merge_dur):
    args.test_merge_dur = str(args.merge_dur)
val_loaders = {}

for test_merge_dur in args.test_merge_dur.split(','):
    test_merge_dur = int(float(test_merge_dur))
    if args.mode == 'eval_gold':
        val_loaders[test_merge_dur] = torch.utils.data.DataLoader(
            dataloader_fhs_gold.FHSGoldDataset(manifest_dir=args.manifest_dir, metadata_path=args.metadata_path, split='test', audio_conf=val_audio_conf, n_class=args.n_class, merge_dur=test_merge_dur),
            batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        val_loaders[test_merge_dur] = torch.utils.data.DataLoader(
            dataloader_fhs_10khr.AudiosetDataset(manifest_dir=args.manifest_dir, metadata_path=args.metadata_path, split='test', audio_conf=val_audio_conf, n_class=args.n_class, merge_dur=test_merge_dur, num_all_per_class=args.num_all_per_class, add_silence=args.add_silence, speakers=args.speakers.split(','), diarize=args.diarize),
            batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
print('Test segment duration:', args.test_merge_dur)

if args.mode == 'train':
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader_fhs_10khr.AudiosetDataset(manifest_dir=args.manifest_dir, metadata_path=args.metadata_path, split='train', audio_conf=audio_conf, n_class=args.n_class, merge_dur=args.merge_dur, num_all_per_class=args.num_all_per_class, add_silence=args.add_silence, speakers=args.speakers.split(','), diarize=args.diarize),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


if args.model == 'ast':
    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size=args.model_size,pos_emb_type=args.pos_emb_type)
elif args.model == 'DASS':
    print('Using DASS model')
    if args.loss == 'BCE':
        audio_model = models.DASS(label_dim=1, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size=args.model_size)
    else:
        audio_model = models.DASS(label_dim=args.n_class, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size=args.model_size)


print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir, exist_ok=True)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

teacher_model = None
if args.knowledge_distillation:
    if args.kd_teach_type == 'ast':
        print('Using knowledge distillation from AST with loss type: {:s} loss temp {:.2f} loss weight {:.2f}'.format(args.dist_loss_type, args.dist_temp,args.dist_loss_weight) )
        teacher_model = models.ASTModel(label_dim=args.n_class, fstride=10, tstride=10, input_fdim=128,
                                        input_tdim=args.audio_length, imagenet_pretrain=True,
                                        audioset_pretrain=True, model_size='base384')

        if os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth') == False:
            import wget
            audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
            wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
        sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location='cpu')
        if not isinstance(teacher_model, torch.nn.DataParallel):
            teacher_model = torch.nn.DataParallel(teacher_model)
        teacher_model.load_state_dict(sd, strict=True)
    elif args.kd_teach_type == 'DASS':
        print('Using knowledge distillation from DASS with loss type: {:s} loss temp {:.2f} loss weight {:.2f}'.format(args.dist_loss_type, args.dist_temp,args.dist_loss_weight))
        ## tiny mast model with 47.1 mAP
        if args.kd_teach_dir == '':
            print("Please specify the teacher model directory for knowledge distillation")
            exit()
        if 'small' in args.kd_teach_dir:
            model_size = 'small'
        elif 'medium' in args.kd_teach_dir:
            model_size = 'medium'
        teacher_model = models.DASS(label_dim=args.n_class, imagenet_pretrain=True,
                                        audioset_pretrain=args.audioset_pretrain, model_size=model_size)
        sd = torch.load(args.kd_teach_dir, map_location='cpu')
        if not isinstance(teacher_model, torch.nn.DataParallel):
            teacher_model = torch.nn.DataParallel(teacher_model).eval()
        teacher_model.load_state_dict(sd, strict=True)
    elif args.kd_teach_type == 'self':
        import copy
        print('Using knowledge distillation from self with loss type: {:s} loss temp {:.2f} loss weight {:.2f}'.format(args.dist_loss_type, args.dist_temp,args.dist_loss_weight))
        teacher_model = copy.deepcopy(audio_model)
        
        if not isinstance(teacher_model, torch.nn.DataParallel):
            teacher_model = torch.nn.DataParallel(teacher_model).eval()

if args.mode == 'train':
    print('Now starting training for {:d} epochs'.format(args.n_epochs))
    train(audio_model,teacher_model, train_loader, val_loaders, args)
else:
    #try:
    #    audio_model = torch.nn.DataParallel(audio_model)
    #    audio_model.load_state_dict(torch.load(os.path.join(args.exp_dir, 'models/best_audio_model.pth')))
    #except:
    audio_model.load_state_dict(torch.load(os.path.join(args.exp_dir, 'models/best_audio_model.pth')))
    stats, loss = validate(audio_model, val_loaders, args, -1)
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
        'Test Segment Duration',
        'Best Macro F1',
        'Best AUC',
    ]
    result = {k:[] for k in header}
    for merge_dur in sorted(stats):
        best_auc = 0
        best_f1 = 0
        for topk in sorted(stats[merge_dur]):
            p = stats[merge_dur][topk]['p']
            r = stats[merge_dur][topk]['r']
            acc = stats[merge_dur][topk]['acc']
            f1 = stats[merge_dur][topk]['f1']
            auc = stats[merge_dur][topk]['auc']
            if auc > best_auc:
                best_auc = auc

            if f1 > best_f1:
                best_f1 = stats[merge_dur][topk]['f1']

            print(f'Top {topk}, Precision: {p:.3f}, Recall: {r:.3f}, Accuracy: {acc:.3f}, Macro F1: {f1:.3f}, AUC: {auc:.3f}')
            result['Epoch'].append(0)
            result['Step'].append(0)
            result['Top k'].append(topk)
            result['Train Loss'].append(loss)
            result['Test Segment Duration'].append(merge_dur)
            result['Test Precision'].append(p)
            result['Test Recall'].append(r)
            result['Test Micro F1'].append(acc)
            result['Test Macro F1'].append(f1)
            result['Test AUC'].append(auc)
            result['Best AUC'].append(best_auc)
            result['Best Macro F1'].append(best_f1)

    df = pd.DataFrame(result)
    df.to_csv(args.exp_dir + '/test_result.csv')
