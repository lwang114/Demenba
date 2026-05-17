#!/bin/bash
#SBATCH -p 2080
##SBATCH -p a6
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:4
#SBATCH -n 1
#SBATCH -J text_fhs
#SBATCH -o logs/%j_text_fhs.out
#SBATCH -e logs/%j_text_fhs.err

export HOME=/data/sls/scratch/limingw
CONDA_ROOT=$HOME/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /data/sls/scratch/limingw/miniconda3/envs/nemo
cd ..

model_suffix=Llama-3.1-8B-Instruct
model_name=meta-llama/$model_suffix
llama_factory_path=/data/sls/scratch/limingw/workplace/LLaMA-Factory
manifest_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs

n_parts=$1
i=$2
s=0
e=10
if [ $s -le 0 ] && [ $e -ge 0 ]; then
#    for x in train_800 test_100; do
    for x in train_1200; do
        echo $x
# XXX   for x in test_1200 test_150; do
        python steps/diarize.py \
            --tsv-path $manifest_dir/$x.tsv \
            --out-path $manifest_dir/${x}_diarized.tsv
    done
fi

if [ $s -le 1 ] && [ $e -ge 1 ]; then
# XXX    for x in train_800; do
    for x in train_1200; do
        python utils/split_tsv.py \
            --tsv-path $manifest_dir/${x}_diarized.tsv \
            --n-parts $n_parts \
            --out-prefix $manifest_dir/${x}_diarized_part
    done
fi

if [ $s -le 10 ] && [ $e -ge 10 ]; then
    conda activate /data/sls/scratch/limingw/miniconda3/envs/whisper
#    for x in test_150 test_100; do
#        python steps/faster_whisper_segment_asr.py \
#            --tsv_path $manifest_dir/${x}_diarized.tsv \
#            --out_path $manifest_dir/${x}_whisper.wrd
#    done

# XXX  for x in train_800; do
    for x in train_1200; do
        echo $x
        python steps/faster_whisper_segment_asr.py \
        --tsv_path $manifest_dir/${x}_diarized_part_${i}.tsv \
        --out_path $manifest_dir/${x}_whisper_part_${i}.wrd
    done
    conda deactivate
fi

if [ $s -le 11 ] && [ $e -ge 11 ]; then
# XXX   for x in train_800; do
    for x in train_1200; do
        python utils/combine_parts.py \
            --in_path $manifest_dir/${x}_whisper_part \
            --n_parts $n_parts \
            --out_path $manifest_dir/${x}_whisper.wrd
    done
fi

if [ $s -le 12 ] && [ $e -ge 12 ]; then
    for x in train_1200; do
# XXX   for x in train_800 test_100; do
        python utils/combine_diarize_and_asr_outputs_v2.py \
            --diarize-path $manifest_dir/${x}_diarized.tsv \
            --asr-path $manifest_dir/${x}_whisper.wrd \
            --out-path $manifest_dir/${x}_diarized_whisper.wrd
    done
fi

if [ $s -le 20 ] && [ $e -ge 20 ]; then
    for x in train_1200; do
# XXX   for x in train_800 test_100; do
        python utils/convert_to_llama_factory_format.py \
            --wrd-path $manifest_dir/${x}_diarized_whisper.wrd \
            --tsv-path $manifest_dir/$x.tsv \
            --lbl-path $manifest_dir/$x.lbl \
            --out-path data/${x}.json \
            --label-set normal,mild,dementia
# XXX            --label-set normal,dementia
    done
fi

if [ $s -le 30 ] && [ $e -ge 30 ]; then
    conda activate llama_factory
    cwd=$PWD
    cd $llama_factory_path
# XXX   for x in train_800 test_100; do
    for x in train_1200; do
        cp data/${x}.json $llama_factory_path/data/fhs_$x.json
        llamafactory-cli train \
            --model_name_or_path $model_name \
            --stage sft \
            --do_train true \
            --finetuning_type lora \
            --lora_target all \
            --dataset fhs_$x \
            --template llama3 \
            --cutoff_len 1024 \
            --max_samples 10000 \
            --overwrite_cache true \
            --preprocessing_num_workers 16 \
            --output_dir saves/${model_suffix}_fhs_$x \
            --logging_steps 10 \
            --save_steps 50 \
            --plot_loss true \
            --overwrite_output_dir true \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-4 \
            --num_train_epochs 20 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --bf16 true \
            --ddp_timeout 180000000 \
            --val_size 0.1 \
            --per_device_eval_batch_size 1 \
            --eval_strategy steps \
            --eval_steps 500
    done
    cd $cwd
    conda deactivate
fi

if [ $s -le 40 ] && [ $e -ge 40 ]; then
    x=test_100
    merge_dur=360
    conda activate llama_factory
#    for step in 640 600 550 500 450 400 350 300 250 200 150 100; do
    for step in 300; do
        echo Step: $step
        peft_dir=$llama_factory_path/saves/${model_suffix}_fhs_2class_800/checkpoint-$step 
        python steps/llama_ft_eval.py \
            --peft_model_dir $peft_dir \
            --tsv_path $manifest_dir/$x.tsv \
            --label_path $manifest_dir/$x.lbl \
            --label_set normal,dementia \
            --text_path $manifest_dir/${x}_diarized_whisper.wrd \
            --out_path $peft_dir/../$x \
            --prompt_type simple \
            --merge_dur $merge_dur
    done 
    conda deactivate
fi
