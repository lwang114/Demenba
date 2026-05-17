#!/bin/bash
#SBATCH -p a5
##SBATCH -p a6
##SBATCH -p 2080
#SBATCH --qos=regular
##SBATCH --gres=gpu:2
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -J mast_as_small_fhs_3class
#SBATCH -o logs/%j_mast_as_small_fhs_3class.out
#SBATCH -e logs/%j_mast_as_small_fhs_3class.err

# set -x
# comment this line if not running on sls cluster
export HOME=/data/sls/scratch/limingw
CONDA_ROOT=$HOME/miniconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate vmamba
which python
export TORCH_HOME=../../pretrained_models

model=DASS
#model_size=medium
model_size=small
dataset=audioset
# full or balanced for audioset
set=balanced
imagenetpretrain=True
#imagenetpretrain=False
audiosetpretrain=False
#audiosetpretrain=True
knowledge_distillation=False
dist_loss_type=kldiv
dist_temp=1.0
if [ $set == balanced ]
then
  bal=none
  lr=0.0001
  epoch=40
  lrscheduler_start=10
  lrscheduler_step=5
  lrscheduler_decay=0.5
  wa_start=6
  wa_end=25
else
  bal=bal
  lr=0.0001
  epoch=5
  lrscheduler_start=2
  lrscheduler_step=1
  lrscheduler_decay=0.5
  wa_start=6
  wa_end=25
fi
freqm=48
timem=192
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=10
tstride=10

dataset_mean=-4.2677393
dataset_std=4.5689974

merge_dur=$1
if [ $merge_dur == 720 ]; then
    batch_size=1
    audio_length=73728
elif [ $merge_dur == 540 ]; then
    batch_size=1
    audio_length=69120
elif [ $merge_dur == 450 ]; then
    batch_size=1
    audio_length=46080
elif [ $merge_dur == 360 ]; then
    batch_size=1
    audio_length=36864
elif [ $merge_dur == 180 ]; then
    batch_size=1
    audio_length=18432
elif [ $merge_dur == 90 ]; then
    batch_size=2
    audio_length=9216
else
    batch_size=2
    audio_length=3072
fi

#num_all_per_class=1000
num_all_per_class=$2
n_class=3
lr=$3
noise=False
#add_silence=False
add_silence=$4
mode=$5

metrics=f1
loss=$6
vote=$7
test_merge_dur=$8
diarize=$9

warmup=True
wa=False

if [ $model == DASS ]
then
exp_dir=./exp_fhs_merge-dur${merge_dur}_${n_class}class_num-all-per-class${num_all_per_class}/${model}-${model_size}-${set}-p$imagenetpretrain-ap-b$batch_size-lr${lr}-kd${knowledge_distillation}-kdd${dist_loss_type}-dt${dist_temp}-add_silence_${add_silence}-diarize_${diarize}-loss_${loss}
#exp_dir=./exp_fhs_merge-dur${merge_dur}_${n_class}class_num-all-per-class${num_all_per_class}/${model}-${model_size}-${set}-p$imagenetpretrain-ap$audiosetpretrain-b$batch_size-lr${lr}-kd${knowledge_distillation}-kdd${dist_loss_type}-dt${dist_temp}-add_silence_${add_silence}-loss_${loss}
#exp_dir=./exp_fhs_merge-dur${merge_dur}_num-all-per-class${num_all_per_class}/${model}-${model_size}-${set}-p$imagenetpretrain-b$batch_size-lr${lr}-kd${knowledge_distillation}-kdd${dist_loss_type}-dt${dist_temp}
fi

mkdir -p $exp_dir
echo $exp_dir

s=0
e=0
if [ $s -le 0 ] && [ $e -ge 0 ]; then
    CUDA_CACHE_DISABLE=1 python -W ignore src/run_fhs.py --model ${model} --model_size ${model_size} --dataset ${dataset} --merge_dur $merge_dur \
    --test_merge_dur ${test_merge_dur} --vote $vote \
    --n_class $n_class --num_all_per_class $num_all_per_class \
    --exp-dir $exp_dir --audioset_pretrain $audiosetpretrain \
    --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
    --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
    --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
    --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
    --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} \
    --knowledge_distillation ${knowledge_distillation} --dist_loss_type ${dist_loss_type} --dist_temp ${dist_temp} \
    --add_silence $add_silence --diarize $diarize --mode $mode
fi

if [ $s -le 1 ] && [ $e -ge 1 ]; then
#    for spk in Participant; do
#    for spk in Interviewer; do
    for spk in Participant Interviewer; do
        echo "Keep $spk segment only"
        exp_dir=./exp_fhs_merge-dur${merge_dur}_${n_class}class_num-all-per-class${num_all_per_class}/${model}-${model_size}-${set}-p$imagenetpretrain-ap-b$batch_size-lr${lr}-kd${knowledge_distillation}-kdd${dist_loss_type}-dt${dist_temp}-add_silence_${add_silence}-diarize_${diarize}-loss_${loss}-${spk}_only
        CUDA_CACHE_DISABLE=1 python -W ignore src/run_fhs.py --model ${model} --model_size ${model_size} --dataset ${dataset} --merge_dur $merge_dur \
        --test_merge_dur ${test_merge_dur} --vote $vote \
        --n_class $n_class --num_all_per_class $num_all_per_class \
        --exp-dir $exp_dir --audioset_pretrain $audiosetpretrain \
        --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
        --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
        --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
        --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
        --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
        --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} \
        --knowledge_distillation ${knowledge_distillation} --dist_loss_type ${dist_loss_type} --dist_temp ${dist_temp} \
        --add_silence $add_silence --diarize $diarize --mode $mode \
        --speakers $spk
    done
fi

