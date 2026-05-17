import argparse
from collections import defaultdict
import numpy as np
import random
from sklearn.metrics import roc_auc_score 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from pathlib import Path
import torch
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")

def create_prompt(text, label_set, prompt_type='simple'):
    if prompt_type == 'simple':
        prompt = (
            "You are a helpful assistant that classifies if a participant in an interview has dementia.\n\n"
            f"Possible labels: {', '.join(label_set)}.\n"
            f"Interview transcript: \"{text}\"\n"
            "Label:"
        )
    elif prompt_type == 'cot':
        prompt = (
            "You are a helpful assistant that classifies if a participant in an interview has dementia.\n\n"
            f"Possible labels: {', '.join(label_set)}.\n"
            f"Interview transcript: \"{text}\"\n"
            "Answer: let's think step by step."
        )
    else: raise ValueError(f'Unknown prompt type: {prompt_type}')
    
    return prompt

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

def majority_vote(pred_labels, pred_scores, label_set):
    n_class = len(label_set)
    prob = np.zeros(n_class)
    poll = np.zeros(n_class+1)
    for y, p in zip(pred_labels, pred_scores):
        prob += p
        if y not in label_set:
            poll[-1] += 1
        else:
            poll[label_set.index(y)] += 1

    y = poll.argmax()
    if y == len(label_set):
        return 'unknown', prob / len(pred_labels)
    else:
        return label_set[y], prob / len(pred_labels)

def classify(model, tokenizer, text, label_set, args):
    prompt = create_prompt(text, label_set)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits  # [batch_size, seq_len, vocab_size]

    # === Get logits of the last token ===
    last_token_logits = logits[0, -1]  # shape: [vocab_size]
    label_ids = tokenizer(label_set, add_special_tokens=False).input_ids
    label_ids = [lab[0] for lab in label_ids]
    kept_last_logits = last_token_logits[label_ids]
    probs = kept_last_logits.softmax(-1).detach().cpu().numpy()

    # === Decode top-5 tokens for inspection ===
    topk = torch.topk(last_token_logits, k=3)
    tokens = tokenizer.convert_ids_to_tokens(topk.indices.tolist())
    label = tokens[0]
    top_probs = torch.nn.functional.softmax(topk.values, dim=-1).detach().cpu().numpy()

#    for token, top_prob in zip(tokens, top_probs):
#        print(f"{token}: {top_prob.item():.4f}")
    
    return label, probs, ''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--peft_model_dir', default='/data/sls/scratch/limingw/workplace/LLaMA-Factory/saves/Llama-3.1-8B-Instruct_fhs_2class_800/checkpoint-400')
    parser.add_argument('--tsv_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100.tsv')
    parser.add_argument('--label_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100.lbl')
    parser.add_argument('--label_set', default='normal,dementia')
    parser.add_argument('--text_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100_diarized_whisper.wrd')
    parser.add_argument('--out_path', default='/data/sls/scratch/limingw/workplace/LLaMA-Factory/saves/Llama-3.1-8B-Instruct_fhs_2class_800/test_100')
    parser.add_argument('--prompt_type', default='simple')
    parser.add_argument('--merge_dur', type=int, default=360)
    return parser

def main():
    set_seed()
    parser = get_parser()
    args = parser.parse_args()
    Path(args.out_path).parent.mkdir(exist_ok=True, parents=True)
    out_label_path = args.out_path+'.lbl'
    out_gold_label_path = args.out_path+'_gold.lbl'
    out_explanation_path = args.out_path+'_explanation.lbl'
    out_score_path = args.out_path+'.npy'
    out_gold_onehot_path = args.out_path+'_gold_onehot.npy'
    out_res_path = args.out_path+'_result.txt'
    label_set = args.label_set.split(',')

    # === Step 1: Load PEFT config ===
    peft_model_dir = args.peft_model_dir  # Path to your LLaMA-Factory checkpoint
    peft_config = PeftConfig.from_pretrained(peft_model_dir)

    # === Step 2: Load base model and tokenizer ===
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Step 3: Load LoRA adapter ===
    model = PeftModel.from_pretrained(model, peft_model_dir)
    model.eval()

    # === Step 4: Prepare input and get logits ===
    fns = []
    with open(args.tsv_path, 'r') as f_tsv,\
        open(args.text_path, 'r') as f,\
        open(args.label_path, 'r') as f_lab:
        lines = f_tsv.read().strip().split('\n')
        _ = lines.pop(0)
        texts = f.read().strip().split('\n')
        labels = f_lab.read().strip().split('\n')
        wav2seg = defaultdict(list)
        wav2lab = {}
        for l, y in tqdm(list(zip(lines, labels))):
            fn = l.strip().split('\t')[0]
            if not fn in fns:
                fns.append(fn)
                wav2lab[fn] = label_set[int(y)]

        for l in tqdm(texts):
            fn, start, end = l.strip().split('\t')[:3]
            text = ''.join(l.strip().split('\t')[3:])
            wav2seg[fn].append([int(start), int(end), text])
        wav2text = merge(wav2seg, args.merge_dur)

    pred_scores = []
    pred_labels = []
    gold_labels = []
    with open(out_label_path, 'w') as f_out,\
        open(out_gold_label_path, 'w') as f_out_gold,\
        open(out_explanation_path, 'w') as f_out_exp,\
        open(out_res_path, 'a') as f_out_res:
        for fn in tqdm(fns):
            labels = []
            scores = [] 
            explanations = []
            for start, end, text in wav2text[fn]:
                label, score, explanation = classify(model, tokenizer, text, label_set, args)
                labels.append(label)
                scores.append(score)
                explanations.append(explanation)

            print(f'{fn}\t{",".join(labels)}', file=f_out)
            print(f'{fn}\t{wav2lab[fn]}', file=f_out_gold)
            print(f'{fn}\t{" ".join(explanations)}', file=f_out_exp)
            label, score = majority_vote(labels, np.asarray(scores), label_set)
            print(f'pred: {label}, gold: {wav2lab[fn]}', flush=True)
            pred_labels.append(label)
            gold_labels.append(wav2lab[fn])
            pred_scores.append(score)
        correct = [y_p == y for y_p, y in zip(pred_labels, gold_labels)]
        acc = np.asarray(correct).mean()
        print(f'Accuracy: {acc}')
        print(f'Accuracy: {acc}', file=f_out_res)
        
        gold_labels_onehot = [np.eye(len(label_set))[label_set.index(y)] for y in gold_labels]
        gold_labels_onehot = np.stack(gold_labels_onehot)
        scores = np.stack(pred_scores)
        np.save(out_gold_onehot_path, gold_labels_onehot)
        np.save(out_score_path, scores)

        n_class = scores.shape[-1]
        auc = np.mean(
            [roc_auc_score(gold_labels_onehot[:, i], scores[:, i]) for i in range(n_class)]
        )
        print(f'AUC: {auc}')
        print(f'AUC: {auc}', file=f_out_res)

if __name__ == '__main__':
    main()
