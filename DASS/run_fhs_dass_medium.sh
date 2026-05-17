#!/bin/bash

#manifest_dir=/path/to/your/data
#metadata=/path/to/your/metadata
#exp_dir=/path/to/your/experiment
manifest_dir=/orcd/home/002/limingw/orcd/pool/workplace/fhs_dementia_detector/data/test
metadata_path=/orcd/home/002/limingw/orcd/pool/workplace/fhs_dementia_detector/data/fhs/add_cog_data_\(5586\)_\[11717\]_20250131_14_27_9_0179_dvoice_and_npath_add_dr.csv
exp_dir
exp_dir=./exp_dass_medium
sbatch run_fhs_dass_medium.slurm 360 400 0.0001 True train WCE $manifest_dir $metadata_path $exp_dir
# bash run_fhs_dass_medium.slurm 360 400 0.0001 True train WCE $manifest_dir $metadata_path $exp_dir
