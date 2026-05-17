#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=8
#SBATCH -J fhs_10khr_text_classifier
#SBATCH -o logs/%j_fhs_10khr_text_classifier.out
#SBATCH -e logs/%j_fhs_10khr_text_classifier.err 

export HOME=/orcd/home/002/limingw
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate DASS
cd ..

lm_name=$1
classifier=$2
add_silence=$3
num_train=$4
merge_dur=$5
n_class=$6
tgt_dir=$7
#tgt_dir=$PWD/../data/fhs
metadata_path=$8
exp_dir=$9
mode=${10}
#exp_dir=$PWD/exp/${classifier}-${lm_name}-${n_class}class-${num_train}hr-sorted_addsil_${add_silence}_loss-$loss

if [ $n_class = 1 ]; then
    loss=bce
else
    loss=wce
fi
#loss=wce
#exp_dir=$PWD/exp/${classifier}_${feat_name}_${num_train}hr_sorted_addsil$add_silence
echo $exp_dir
#exp_dir=$PWD/exp/debug  # XXX
stage=0
stop_stage=0
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    cp $tgt_dir/test_100_diarized_whisper.wrd $tgt_dir/test_800_diarized_whisper.wrd
    cp $tgt_dir/test_100_diarized_whisper.wrd $tgt_dir/test_1200_diarized_whisper.wrd 
    python traintest_text.py \
        --data-dir $tgt_dir \
	--metadata-path $metadata_path \
        --exp-dir $exp_dir \
        --lm $lm_name \
        --classifier $classifier \
	--layers 0 \
        --batch_size 1 \
        --add-silence $add_silence \
        --n_class $n_class \
        --num-train $num_train \
        --merge-dur $merge_dur \
        --loss $loss \
	--mode $mode
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    tgt_dir=$PWD/../data/fhs_gold92
    python traintest_text.py \
        --data-dir $tgt_dir \
        --exp-dir $exp_dir \
        --lm $lm_name \
        --classifier $classifier \
        --layers 0 \
        --batch_size 8 \
        --add-silence $add_silence \
        --n_class $n_class \
        --num-train $num_train \
        --merge-dur $merge_dur \
        --loss $loss \
        --mode eval_gold \
        --text-type whisper
fi
