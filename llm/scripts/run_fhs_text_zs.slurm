#!/bin/bash
#SBATCH -p a6
#SBATCH --qos=regular
##SBATCH --gres=gpu:3
#SBATCH --gres=gpu:2
##SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -J text_preprocess_fhs
#SBATCH -o logs/%j_text_preprocess_fhs.out
#SBATCH -e logs/%j_text_preprocess_fhs.err

export HOME=/data/sls/scratch/limingw
CONDA_ROOT=$HOME/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /data/sls/scratch/limingw/miniconda3/envs/nemo
cd ..

manifest_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs
#model_name=meta-llama/Llama-3.1-70B-Instruct
#model_name=Qwen/Qwen2.5-7B-Instruct-1M
model_name=$1
max_new_token=$2
max_new_token=32
merge_dur=$3

s=40
e=50
if [ $s -le 0 ] && [ $e -ge 0 ]; then
    for x in test_100; do
        python steps/diarize.py \
            --tsv-path $manifest_dir/$x.tsv \
            --out-path $manifest_dir/${x}_diarized.tsv
    done
fi

if [ $s -le 10 ] && [ $e -ge 10 ]; then
    conda activate whisper
    for x in test_100; do
        python steps/faster_whisper_segment_asr.py \
            --tsv_path $manifest_dir/${x}_diarized.tsv \
            --out_path $manifest_dir/${x}_whisper.wrd
    done
    conda deactivate
fi

if [ $s -le 20 ] && [ $e -ge 20 ]; then
    for x in test_150; do
        python utils/combine_diarize_and_asr_outputs.py \
            --diarize-path $manifest_dir/${x}_diarized.tsv \
            --asr-path $manifest_dir/${x}_whisper.wrd \
            --out-path $manifest_dir/${x}_diarized_whisper.wrd
    done
fi

if [ $s -le 30 ] && [ $e -ge 30 ]; then
    x=test_150
    prompt_type=cot
    model=$(echo $model_name | cut -d"/" -f 2)
    python steps/llama_zs.py \
        --model_name $model_name \
        --tsv_path $manifest_dir/${x}.tsv \
        --label_path $manifest_dir/${x}.lbl \
        --label_set normal,mild,dementia \
        --text_path $manifest_dir/${x}_diarized_whisper.wrd \
        --out_path ./exp/${x}_${model}_${prompt_type} \
    	--prompt_type $prompt_type
fi

if [ $s -le 40 ] && [ $e -ge 40 ]; then
    x=test_100
    for prompt_type in simple; do  #cot
        model=$(echo $model_name | cut -d"/" -f 2)
        python steps/llama_zs.py \
            --model_name $model_name \
            --tsv_path $manifest_dir/${x}.tsv \
            --label_path $manifest_dir/${x}.lbl \
            --label_set normal,dementia \
            --text_path $manifest_dir/${x}_diarized_whisper.wrd \
            --out_path ./exp/${x}_${model}_${prompt_type} \
            --prompt_type $prompt_type \
            --max_new_token $max_new_token
    done
fi

if [ $s -le 50 ] && [ $e -ge 50 ]; then
#    model_name=meta-llama/Llama-3.1-70B-Instruct
#    model_name=meta-llama/Llama-3.1-8B-Instruct
    x=test_100
    for prompt_type in simple; do  #cot
        model=$(echo $model_name | cut -d"/" -f 2)
        python steps/llama_zs_majority_vote.py \
            --model_name $model_name \
            --tsv_path $manifest_dir/${x}.tsv \
            --label_path $manifest_dir/${x}.lbl \
            --label_set normal,dementia \
            --text_path $manifest_dir/${x}_diarized_whisper.wrd \
            --out_path ./exp/${x}_${model}_${prompt_type}_mergedur${merge_dur} \
            --prompt_type $prompt_type \
            --max_new_token $max_new_token \
            --merge_dur $merge_dur
    done
fi
