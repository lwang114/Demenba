#!/bin/bash
##SBATCH -p a5
#SBATCH -p a6
##SBATCH -p 2080
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -J mast_as_medium_fhs
#SBATCH -o logs/%j_mast_as_medium_fhs_3class.out
#SBATCH -e logs/%j_mast_as_medium_fhs_3class.err

# set -x
# comment this line if not running on sls cluster
export HOME=/data/sls/scratch/limingw
CONDA_ROOT=$HOME/miniconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate vmamba
which python
export TORCH_HOME=../../pretrained_models

model=DASS
model_size=medium
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
  lr=0.00005
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
test_merge_dur=$1
if [ $merge_dur == 360 ]; then
    batch_size=1
    audio_length=36864
elif [ $merge_dur == 180 ]; then
    batch_size=1
    audio_length=18432
elif [ $merge_dur == 90 ]; then
    batch_size=1
    audio_length=9216
else
    batch_size=1
    audio_length=3072
fi

#num_all_per_class=1000
num_all_per_class=$2
lr=$3
n_class=3
noise=False
#add_silence=False
add_silence=$4
mode=$5
loss=$6
manifest_dir=$7
metadata_path=$8
metrics=f1
warmup=True
wa=False
diarize=True

exp_dir=$9

mkdir -p $exp_dir
echo $exp_dir

s=0
e=0

CUDA_CACHE_DISABLE=1 python -W ignore src/run_fhs.py --model ${model} --model_size ${model_size} --dataset ${dataset} --merge_dur $merge_dur \
--manifest_dir $manifest_dir \
--metadata_path $metadata_path \
--dataset ${dataset} --merge_dur $merge_dur \
--n_class $n_class --num_all_per_class $num_all_per_class \
--exp-dir $exp_dir --audioset_pretrain $audiosetpretrain \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} \
--knowledge_distillation ${knowledge_distillation} --dist_loss_type ${dist_loss_type} --dist_temp ${dist_temp} \
--add_silence $add_silence --mode $mode --diarize $diarize
