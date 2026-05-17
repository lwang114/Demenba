#!/bin/bash
#SBATCH -J llama_ft
#SBATCH -o logs/%j_llama_ft.out
#SBATCH -e logs/%j_llama_ft.err
##SBATCH --qos=priority
#SBATCH --gres=gpu:4
#SBATCH --qos=regular
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=a6


export HOME=/data/sls/scratch/limingw
export CONDA_ROOT=/data/sls/scratch/limingw/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate llama_factory

model_path=meta-llama/Meta-Llama-3.1-8B-Instruct
vocab_size=1000
manifest_dir=data/flickr8k_top$vocab_size

stage=20
stop_stage=20
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
#    n_shot=100
    min_shift=6
    max_shift=15
    max_token_len=2048
    sweep_hp='n_shot'
    if [ $sweep_hp = 'n_shot' ]; then
        for n_shot in 5 10 15 20; do
            python scripts/create_synthetic_unseen_language_llm_finetune.py \
                --manifest-dir $manifest_dir \
                --out-dir LLaMA-Factory/data/flickr8k_text_only_${min_shift}_${max_shift} \
                --n-shots $n_shot \
                --min-shift $min_shift \
                --max-shift $max_shift
            #python scripts/convert_llama_llamafactory.py \
            #    --in-path ${manifest_dir}_shift5/${n_shot}shot/test.json \
            #    --out-path LLaMA-Factory/data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot/test.json
            for x in train; do
                python scripts/cut_prompts.py \
                    --in-path LLaMA-Factory/data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot/${x}.json \
                    --out-path LLaMA-Factory/data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot/${x}.json \
                    --max-token-len $max_token_len
            done
        done
    else
        n_shot=5
        for max_shift in 15 20 25; do
            python scripts/create_synthetic_unseen_language_llm_finetune.py \
                --manifest-dir $manifest_dir \
                --out-dir LLaMA-Factory/data/flickr8k_text_only_${min_shift}_${max_shift} \
                --n-shots $n_shot \
                --min-shift $min_shift \
                --max-shift $max_shift
            #python scripts/convert_llama_llamafactory.py \
            #    --in-path ${manifest_dir}_shift5/${n_shot}shot/test.json \
            #    --out-path LLaMA-Factory/data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot/test.json
            for x in train; do
                python scripts/cut_prompts.py \
                    --in-path LLaMA-Factory/data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot/${x}.json \
                    --out-path LLaMA-Factory/data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot/${x}.json \
                    --max-token-len $max_token_len
            done
        done
    fi
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    n_shots=(10 15 20) 
    cwd=$PWD
    cd LLaMA-Factory
    min_shift=6
    max_shift=15
    max_token_len=2048
    sweep_hp='n_shot'
    if [ $sweep_hp = 'n_shot' ]; then
        for n_shot in ${n_shots[@]}; do
            if [ -d ./data/flickr8k_text_only/0shot ]; then
                rm -r ./data/flickr8k_text_only/0shot
            fi
            cp -r ./data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot ./data/flickr8k_text_only/0shot 
            
            llamafactory-cli train \
                --model_name_or_path $model_path \
                --stage sft \
                --do_train true \
                --finetuning_type lora \
                --lora_target all \
                --dataset flickr8k_text_only_0shot \
                --template llama3 \
                --cutoff_len $max_token_len \
                --max_samples 10000 \
                --overwrite_cache true \
                --preprocessing_num_workers 16 \
                --output_dir saves/llama3-8b/lora/sft/llama3_1-8b_flickr8k_text_only_${min_shift}_${max_shift}_${n_shot}shot/lora/sft \
                --logging_steps 10 \
                --save_steps 50 \
                --plot_loss true \
                --overwrite_output_dir true \
                --per_device_train_batch_size 1 \
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
    else
        n_shot=5
        for max_shift in 7 8 9 10; do
            if [ -d ./data/flickr8k_text_only/0shot ]; then
                rm -r ./data/flickr8k_text_only/0shot
            fi
            cp -r ./data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot ./data/flickr8k_text_only/0shot 
            llamafactory-cli train \
                --model_name_or_path $model_path \
                --stage sft \
                --do_train true \
                --finetuning_type lora \
                --lora_target all \
                --dataset flickr8k_text_only_0shot \
                --template llama3 \
                --cutoff_len $max_token_len \
                --max_samples 10000 \
                --overwrite_cache true \
                --preprocessing_num_workers 16 \
                --output_dir saves/llama3-8b/lora/sft/llama3_1-8b_flickr8k_text_only_${min_shift}_${max_shift}_${n_shot}shot/lora/sft \
                --logging_steps 10 \
                --save_steps 50 \
                --plot_loss true \
                --overwrite_output_dir true \
                --per_device_train_batch_size 1 \
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
    fi
    cd $cwd
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    cwd=$PWD
    cd LLaMA-Factory

    n_shots=(5) #(5 10 15 20)
    min_shift=6
    max_shift=15
    max_token_len=2048
    sweep_hp='n_shot'
    if [ $sweep_hp = 'n_shot' ]; then
        for n_shot in ${n_shots[@]}; do
            if [ -d ./data/flickr8k_text_only/0shot ]; then
                rm -r ./data/flickr8k_text_only/0shot
            fi
            # cp -r ./data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot ./data/flickr8k_text_only/0shot
            cp -r ./data/flickr8k_text_only/100shot ./data/flickr8k_text_only/0shot
            # XXX
            llamafactory-cli train \
                --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
                --adapter_name_or_path saves/llama3-8b/lora/sft/llama3_1-8b_flickr8k_text_only_${min_shift}_${max_shift}_${n_shot}shot/lora/sft/checkpoint-3750 \
                --stage sft \
                --do_predict true \
                --finetuning_type lora \
                --eval_dataset flickr8k_text_only_0shot \
                --template llama3 \
                --cutoff_len 1024 \
                --max_samples 50 \
                --overwrite_cache true \
                --preprocessing_num_workers 16 \
                --output_dir saves/llama3-8b/lora/sft/llama3_1-8b_flickr8k_text_only_${min_shift}_${max_shift}_${n_shot}shot/lora/predict_train \
                --overwrite_output_dir true \
                --per_device_eval_batch_size 1 \
                --predict_with_generate true \
                --ddp_timeout 180000000
        done
    else
        n_shot=5
        for max_shift in 7 8 9 10; do
            if [ -d ./data/flickr8k_text_only/0shot ]; then
                rm -r ./data/flickr8k_text_only/0shot
            fi
            cp -r ./data/flickr8k_text_only_${min_shift}_${max_shift}/${n_shot}shot ./data/flickr8k_text_only/0shot
            llamafactory-cli train \
                --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
                --adapter_name_or_path saves/llama3-8b/lora/sft/llama3_1-8b_flickr8k_text_only_${min_shift}_${max_shift}_${n_shot}shot/lora/sft \
                --stage sft \
                --do_predict true \
                --finetuning_type lora \
                --eval_dataset flickr8k_text_only_0shot \
                --template llama3 \
                --cutoff_len 1024 \
                --max_samples 50 \
                --overwrite_cache true \
                --preprocessing_num_workers 16 \
                --output_dir saves/llama3-8b/lora/sft/llama3_1-8b_flickr8k_text_only_${min_shift}_${max_shift}_${n_shot}shot/lora/predict_train \
                --overwrite_output_dir true \
                --per_device_eval_batch_size 1 \
                --predict_with_generate true \
                --ddp_timeout 180000000
        done
    fi
    cd $cwd
fi
