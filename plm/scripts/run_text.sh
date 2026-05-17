#!/bin/bash

manifest_dir=/orcd/home/002/limingw/orcd/pool/workplace/fhs_dementia_detector/data/test
metadata_path=/orcd/home/002/limingw/orcd/pool/workplace/fhs_dementia_detector/data/fhs/add_cog_data_\(5586\)_\[11717\]_20250131_14_27_9_0179_dvoice_and_npath_add_dr.csv
mode=train
exp_dir=./exp_bert_base_cased

for n in 400; do
    for merge_dur in 180; do
        for n_class in 1; do
      	    sbatch run_text.slurm bert-base-cased mlp 1 $n $merge_dur $n_class $manifest_dir $metadata_path $exp_dir $mode
        done
    done
done
