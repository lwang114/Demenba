import argparse
import numpy as np
import pandas as pd
import pathlib
from sklearn.metrics import roc_auc_score 

 
parser = argparse.ArgumentParser()
#parser.add_argument('--audio-score-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-medium-balanced-pTrue-ap-b1-lr0.0001-kdFalse-kddkldiv-dt1.0-add_silence_True-loss_WCE-v2/pred_scores_top3.npy')
#parser.add_argument('--audio-score-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur180_2class_num-all-per-class400/DASS-medium-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE-v2/pred_scores_mergedur180_top1.npy')
#parser.add_argument('--audio-score-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-small-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-loss_WCE/pred_scores_mergedur360_top1.npy')
#parser.add_argument('--audio-score-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_3class_num-all-per-class400/DASS-small-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE/pred_scores_mergedur360_top4.npy')
#parser.add_argument('--audio-score-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-small-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE/eval_gold_pred_scores_mergedur360_top2.npy')
parser.add_argument('--audio-score-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-medium-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE-v2/eval_gold_pred_scores_mergedur360_top4.npy')
#parser.add_argument('--audio-score-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_3class_num-all-per-class400/DASS-medium-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE-v2/pred_scores_mergedur360_top3.npy')
#parser.add_argument('--text-score-path', default='/data/sls/scratch/limingw/workplace/LLaMA-Factory/saves/Llama-3.1-8B-Instruct_fhs_2class_800/test_100.npy')
#parser.add_argument('--text-score-path', default='./exp/mlp-bert-base-cased-1class-400hr-sorted_addsil_1_loss-bce/preds/top2_test_pred_scores.npy')
parser.add_argument('--text-score-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/exp/mlp-bert-base-cased-1class-400hr-sorted_addsil_1_loss-bce/preds/top20_test_pred_scores.npy')
#parser.add_argument('--text-score-path', default='/data/sls/scratch/limingw/workplace/LLaMA-Factory/saves/Qwen2-7B-Instruct_fhs_train_800/test_100.npy')
#parser.add_argument('--true-label-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-medium-balanced-pTrue-ap-b1-lr0.0001-kdFalse-kddkldiv-dt1.0-add_silence_True-loss_WCE-v2/gold_labels_top9.npy')
#parser.add_argument('--true-label-path', default='/data/sls/scratch/limingw/workplace/LLaMA-Factory/saves/Llama-3.1-8B-Instruct_fhs_2class_800/test_100_gold_onehot.npy')
parser.add_argument('--true-label-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-small-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE/eval_gold_gold_labels_mergedur360_top2.npy')
#parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-medium-balanced-pTrue-ap-b1-lr0.0001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE-v2/fusion_result_bert.csv')
#parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur180_2class_num-all-per-class400/DASS-medium-balanced-pTrue-ap-b1-lr0.0001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE-v2/fusion_result_llama.csv')
#parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_3class_num-all-per-class400/DASS-small-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE/fusion_result_bert_2class.tsv')
parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_3class_num-all-per-class400/DASS-small-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE/eval_gold_fusion_result_bert_2class.tsv')
#parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_3class_num-all-per-class400/DASS-medium-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-diarize_True-loss_WCE-v2/fusion_result_bert_2class.tsv')
#parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-medium-balanced-pTrue-ap-b1-lr0.0001-kdFalse-kddkldiv-dt1.0-add_silence_True-loss_WCE-v2/fusion_result_llama.csv')
#parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-small-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-loss_WCE/fusion_result_llama.csv')
#parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_detector/DASS/exp_fhs_merge-dur360_2class_num-all-per-class400/DASS-small-balanced-pTrue-ap-b1-lr0.00001-kdFalse-kddkldiv-dt1.0-add_silence_True-loss_WCE/fusion_result_qwen2.csv')
args = parser.parse_args()

ws = 0.1 * np.arange(11)
audio_scores = np.load(args.audio_score_path)
print(audio_scores.shape)

if audio_scores.shape[-1] > 2:
    audio_scores = np.stack([audio_scores[:, 0], audio_scores[:, 1]+audio_scores[:, 2]], axis=-1)

text_scores = np.load(args.text_score_path)
if len(text_scores.shape) == 1:
    text_scores = np.stack([1-text_scores, text_scores], axis=1)

true_labels = np.load(args.true_label_path)
n_class = text_scores.shape[-1]
if len(true_labels.shape) == 1:
    true_labels = np.eye(n_class)[true_labels]

# text_auc = roc_auc_score(true_labels.flatten(), text_scores.flatten())
audio_auc = np.mean(
    [roc_auc_score(true_labels[:,i], audio_scores[:,i]) for i in range(n_class)]
)
text_auc = np.mean(
    [roc_auc_score(true_labels[:,i], text_scores[:,i]) for i in range(n_class)]
)

print(f'Audio AUC: {audio_auc}, Text AUC: {text_auc}')
df = {r'$\lambda$': [], 'AUC': []}
for w in ws:
    combined_scores = (1 - w) * audio_scores + w * text_scores
    # auc = roc_auc_score(true_labels.flatten(), combined_scores.flatten())
    auc = np.mean(
        [roc_auc_score(true_labels[:,i], combined_scores[:,i]) for i in range(n_class)]
    )
    print(f'Text weight: {w}, AUC: {auc}')
    df[r'$\lambda$'].append(w)
    df['AUC'].append(auc)

df = pd.DataFrame(df)
df.to_csv(args.out_path)
