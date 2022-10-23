#!/bin/bash
# Script to run _locally_  the Deep Speech 2 model training with Common Voice corpus (Mozilla)

# Input data files (CSV and WAV)
data_dir="/data/cv-corpus-8.0-2022-01-19/hu"

# Vocabulary file
voc_file="data/vocabulary-hu.txt"

# Output data
model_dir="model_v0"

# Log file
log_file=log_`date +%Y-%m-%d`

echo "Model training and evaluation..."
start=`date +%s`

nohup python trainer/ds2_train.py --data_dir=$data_dir --vocabulary_file=$voc_file --model_dir=$model_dir --num_gpus=0 --batch_size=64 --seed=1 >$log_file 2>&1&

end=`date +%s`
runtime=$((end-start))
echo "Model training time is" $runtime "seconds."
