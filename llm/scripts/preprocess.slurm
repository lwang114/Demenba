#!/bin/bash
#SBATCH -p a5
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
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
model_name=meta-llama/Llama-3.1-70B-Instruct

s=10
e=20
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
    for x in test_100; do
        python utils/combine_diarize_and_asr_outputs.py \
            --diarize-path $manifest_dir/${x}_diarized.tsv \
            --asr-path $manifest_dir/${x}_whisper.wrd \
            --out-path $manifest_dir/${x}_diarized_whisper.wrd
    done
fi

if [ $s -le 30 ] && [ $e -ge 30 ]; then
    python steps/llama_zs.py \
        --model_name $model_name \
        --label_path $manifest_dir/${x}.lbl \
        --text_path $manifest_dir/${x}_diarized_whisper.wrd \
        --out_path ./exp/test_100_$model_name
fi
