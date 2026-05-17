# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong, Saurabhchand Bhati
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

from collections import defaultdict
import math
import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pandas as pd
import pickle
from torch.cuda.amp import autocast,GradScaler
import copy
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import torch.nn.functional as F
ngpu = torch.cuda.device_count()
print(f'Found {ngpu} CUDA device(s).')


def majority_vote(pred_labels, pred_scores, topk, vote_type='hard'): 
    n_class = pred_scores[0].shape[-1]
    if n_class == 1:
        top_indices = sorted(np.arange(len(pred_scores)), key=lambda x:max(pred_scores[x], 1-pred_scores[x]), reverse=True)[:topk]
        vote = np.zeros(2)
        prob = 0.0
        for i in top_indices:
            vote[int(pred_labels[i])] += 1
            prob += pred_scores[i]
        
        return int(vote.argmax()), prob / len(top_indices)
    else:
        top_indices = sorted(np.arange(len(pred_scores)), key=lambda x:pred_scores[x].max(-1), reverse=True)[:topk]
        n_class = pred_scores[0].shape[-1]
        vote = np.zeros(n_class)
        prob = np.zeros(n_class)
        for k in range(n_class):
            for i in top_indices:
                vote[int(pred_labels[i])] += 1
                prob += pred_scores[i]

        if vote_type == 'hard':
            return int(vote.argmax()), vote / len(top_indices)
        else:
            return int(prob.argmax()), prob / len(top_indices)

def train(audio_model, teach_model,train_loader, test_loader, args):
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
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    distill_loss_meter = AverageMeter()
    teach_loss_meter = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    best_auc, best_f1 = -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    # implementing early stopping
    es_counter = 0
    es_patience = args.es_patience

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_auc, best_f1,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

# XXX   if not isinstance(audio_model, nn.DataParallel):
#        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    if ngpu > 1:
        audio_model.assign_devices(ngpu)

    if teach_model is not None:
# XXX        if not isinstance(teach_model, nn.DataParallel):
#            teach_model = nn.DataParallel(teach_model)
        teach_model = teach_model.to(device)
        teach_model.eval()

    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # dataset specific settings
    # main_metrics = args.metrics
    main_metrics = 'auc'
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == 'WCE':
        if args.n_class == 2:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 3.]).to(torch.float16).to(device))
        else:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 3., 3.]).to(torch.float16).to(device))

    warmup = args.warmup
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    args.loss_fn = loss_fn
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    epoch += 1
    # for amp
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    audio_model.train()
    set_grad_False = False
    #tmp = copy.deepcopy(audio_model.module.v.layers[0].blocks[0].op.in_proj.weight.detach().cpu())
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels, _) in enumerate(train_loader):           
#            if i > 2:  # XXX
#                break
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                audio_output = audio_model(audio_input)
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
                else:
                    loss = loss_fn(audio_output.squeeze(-1), torch.argmax(labels, axis=1).float())

            if teach_model is not None:
                with autocast():
                    with torch.no_grad():
                        teach_output = teach_model(audio_input)
                        teach_loss_meter.update(loss_fn(teach_output, labels).item())

                    if args.dist_loss_type == 'l2c':                    
                        dis_loss = F.mse_loss(audio_output, teach_output) 
                        dis_loss = dis_loss + (1 - torch.sigmoid(F.cosine_similarity(audio_output, teach_output))).mean()
                    elif args.dist_loss_type == 'cosine':
                        dis_loss = 1 - torch.sigmoid(F.cosine_similarity(audio_output, teach_output)).mean()
                    elif args.dist_loss_type == 'l2':
                        dis_loss = F.mse_loss(audio_output, teach_output)
                    elif args.dist_loss_type == 'kldiv':
                        teach_output = teach_output / args.dist_temp
                        dis_loss = F.kl_div(F.log_softmax(audio_output, dim=1), F.log_softmax(teach_output, dim=1), reduction='batchmean',log_target=True)
                    elif args.dist_loss_type == 'bce':
                        if args.dist_teacher_func == 'softmax':
                            teach_output = torch.softmax(teach_output / args.dist_temp, dim=-1)
                        elif args.dist_teacher_func == 'sigmoid':
                            teach_output = torch.sigmoid(teach_output / args.dist_temp)
                        dis_loss = loss_fn(audio_output, teach_output)
                    loss = (1-args.dist_loss_weight)*loss + args.dist_loss_weight*dis_loss 
                    distill_loss_meter.update(dis_loss.item())

            # optimization if amp is not used
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'
                  'Teach Loss {teach_loss_meter.avg:.4f}\t'
                  'Distill Loss {distill_loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter,teach_loss_meter=teach_loss_meter,
                      distill_loss_meter=distill_loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            if teach_model is not None and args.kd_teach_type == 'self':
                with torch.no_grad():
                    m = args.kd_EMA_momentum
                    for param_q, param_k in zip(audio_model.parameters(), teach_model.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            end_time = time.time()
            global_step += 1
            if args.force_limit_batches_per_epoch > 0 and "whole" in args.data_train:
                if i >= args.force_limit_batches_per_epoch * len(train_loader):
                    print("Forcing the epoch %d to end at %d batches" % (epoch, i))
                    break

        print('start validation')
        stats, valid_loss = validate(audio_model, test_loader, args, epoch)
        for merge_dur in sorted(stats):
            print(f'Test segment length: {merge_dur}')
            for topk in sorted(stats[merge_dur]):
                p = stats[merge_dur][topk]['p']
                r = stats[merge_dur][topk]['r']
                acc = stats[merge_dur][topk]['acc']
                f1 = stats[merge_dur][topk]['f1']
                auc = stats[merge_dur][topk]['auc']
                if (merge_dur == args.merge_dur) and (auc > best_auc):
                    best_auc = auc
                    best_epoch = epoch

                if (merge_dur == args.merge_dur) and (f1 > best_f1):
                    best_f1 = stats[merge_dur][topk]['f1']

                print(f'Top {topk}, Precision: {p:.3f}, Recall: {r:.3f}, Accuracy: {acc:.3f}, Macro F1: {f1:.3f}, AUC: {auc:.3f}')
                result['Epoch'].append(epoch)
                result['Step'].append(global_step)
                result['Top k'].append(topk)
                result['Train Loss'].append(loss_meter.avg)
                result['Test Segment Duration'].append(merge_dur)
                result['Test Precision'].append(p)
                result['Test Recall'].append(r)
                result['Test Micro F1'].append(acc)
                result['Test Macro F1'].append(f1)
                result['Test AUC'].append(auc)
                result['Best AUC'].append(best_auc)
                result['Best Macro F1'].append(best_f1)

        df = pd.DataFrame(result)
        df.to_csv(exp_dir + '/result.csv')
        print('validation finished')

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

# XXX       if es_counter >= es_patience and es_patience > 0:
#            print('early stopping after training for {:d} epochs with best AUC of {:.6f}'.format(epoch, best_auc))
#            break

        epoch += 1
        es_counter += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()
        distill_loss_meter.reset()

    if args.wa == True:
        stats = validate_wa(audio_model, test_loader, args, args.wa_start, args.wa_end)
        print('---------------Training Finished---------------')
        print('weighted averaged model results')       
        for topk in sorted(stats):
            auc = stats[topk]['auc']
            p = stats[topk]['p']
            r = stats[topk]['r']
            f1 = stats[topk]['f1']  
            print(f'--------Top {topk} Majority Vote Results--------')
            print("AUC: {:.6f}".format(auc))
            print("Precision: {:.6f}".format(p))
            print("Recall: {:.6f}".format(r))
            print('F1: {:.6f}'.format(f1))
            print("train_loss: {:.6f}".format(loss_meter.avg))
            print("valid_loss: {:.6f}".format(valid_loss))

def validate(audio_model, val_loaders, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
# XXX   if not isinstance(audio_model, nn.DataParallel):
#        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == 'WCE':
        if args.n_class == 2:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 3.]).to(torch.float16).to(device))
        else:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 3., 3.]).to(torch.float16).to(device))

    args.loss_fn = loss_fn

    end = time.time()
    A_scores = defaultdict(list)
    A_predictions = defaultdict(list)
    A_targets = defaultdict(list) 
    A_loss = []
    stats = {merge_dur:defaultdict(dict) for merge_dur in sorted(val_loaders)}
    with torch.no_grad():
        for merge_dur in sorted(val_loaders):
            val_loader = val_loaders[merge_dur]
            for i, (audio_input, labels, id_dates) in enumerate(val_loader):
                audio_input = audio_input.to(device)

                # compute output
                audio_output = audio_model(audio_input)
                if audio_output.shape[-1] > 1:
                    predictions = audio_output.argmax(-1)
                else:
                    predictions = (audio_output > 0).long()

                for logit, label, pred_label, id_date in zip(audio_output, labels, predictions, id_dates):
                    if logit.shape[-1] > 1:
                        A_scores[id_date].append(logit.softmax(-1).detach().cpu().numpy())
                    else:
                        A_scores[id_date].append(logit.sigmoid().detach().cpu().numpy())

                    A_predictions[id_date].append(pred_label.detach().cpu().item())
                    A_targets[id_date] = label.argmax(-1).long().detach().cpu().item()

                # compute the loss
                labels = labels.to(device)
                if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                    loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
                else:
                    loss = args.loss_fn(audio_output.squeeze(-1), torch.argmax(labels.long(), axis=1).float())
                A_loss.append(loss.to('cpu').detach())

                batch_time.update(time.time() - end)
                end = time.time()

            loss = np.mean(A_loss)
            best_auc = 0
            for topk in range(1, 11):
                all_pred_scores = []
                all_pred_labels = []
                all_gold_labels = []
                for id_date, gold_label in sorted(A_targets.items()):
                    if args.vote:
                        pred_label, pred_score = majority_vote(
                            A_predictions[id_date], A_scores[id_date], topk=topk, vote_type=args.vote_type,
                        )
                        all_pred_scores.append(pred_score)
                        all_pred_labels.append(pred_label)
                        all_gold_labels.append(gold_label)
                    else:
                        all_pred_scores.extend(A_scores[id_date])
                        all_pred_labels.extend(A_predictions[id_date])
                        all_gold_labels.extend([gold_label]*len(A_scores[id_date]))

                pred_labels = np.asarray(all_pred_labels)
                pred_scores = np.asarray(all_pred_scores)
                gold_labels = np.asarray(all_gold_labels)

                if args.loss == 'BCE':
                    auc = roc_auc_score(gold_labels, pred_scores)
                    stats[merge_dur][topk]['auc'] = auc
                    fpr, tpr, thresholds = roc_curve(gold_labels, pred_scores)
                    optimal_idx = np.argmax(tpr-fpr)
                    optimal_thres = thresholds[optimal_idx]
                    pred_labels = (pred_scores >= optimal_thres).astype(int)
                    cm = confusion_matrix(gold_labels, pred_labels)
                else:
                    if pred_scores.shape[-1] == 2:
                        fpr, tpr, thresholds = roc_curve(gold_labels, pred_scores[...,1])
                        optimal_idx = np.argmax(tpr-fpr)
                        optimal_thres = thresholds[optimal_idx]
                        pred_labels = (pred_scores[...,1] > optimal_thres)
                    cm = confusion_matrix(gold_labels, pred_labels)
                    gold_labels_onehot = F.one_hot(torch.from_numpy(gold_labels).long(), num_classes=args.n_class).numpy()
                    auc = np.asarray([
                        roc_auc_score(gold_labels_onehot[:,k], pred_scores[:,k])
                        for k in range(args.n_class)
                    ]).mean()
                    stats[merge_dur][topk]['auc'] = auc 

                _, _, acc, _ = precision_recall_fscore_support(
                    gold_labels, pred_labels, average='micro',
                )
                p, r, f1, _ = precision_recall_fscore_support(
                    gold_labels, pred_labels, average='macro',
                )

                if best_auc < auc:
                    best_auc = auc
                    np.save(f'{args.exp_dir}/{args.mode}_pred_scores_mergedur{merge_dur}_top{topk}.npy', pred_scores)
                    np.save(f'{args.exp_dir}/{args.mode}_gold_labels_mergedur{merge_dur}_top{topk}.npy', gold_labels)
                    np.save(f'{args.exp_dir}/{args.mode}_gold_onehot_labels_mergedur{merge_dur}_top{topk}.npy', np.eye(gold_labels.max()+1)[gold_labels])
                    np.save(f'{args.exp_dir}/{args.mode}_confusion_matrix_mergedur{merge_dur}_top{topk}.npy', cm)

                stats[merge_dur][topk]['p'] = p
                stats[merge_dur][topk]['r'] = r
                stats[merge_dur][topk]['acc'] = acc
                stats[merge_dur][topk]['f1'] = f1
            print(merge_dur, stats[merge_dur])
    return stats, loss

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats

def validate_wa(audio_model, val_loader, args, start_epoch, end_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir

    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location=device)

    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        # if choose not to save models of epoch, remove to save space
#        if args.save_model == False:
#            os.remove(exp_dir + '/models/audio_model.' + str(epoch) + '.pth')

    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)

    audio_model.load_state_dict(sdA)

    torch.save(audio_model.state_dict(), exp_dir + '/models/audio_model_wa.pth')

    stats, loss = validate(audio_model, val_loader, args, 'wa')
    return stats
