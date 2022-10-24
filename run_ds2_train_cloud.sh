#!/bin/bash
# Script to run _in Google Cloud_ the Deep Speech 2 model training with Common Voice corpus (Mozilla)

# Bucket for input data files (CSV and WAV)
data_bucket="gs://ds_train/data/cv-corpus-8.0-2022-01-19/hu"

# Vocabulary file
voc_file="data/vocabulary-hu.txt"

# Output data
model_dir="model_v0"

# Log file
log_file=log_`date +%Y-%m-%d`

start=`date +%s`
echo "Model training and evaluation started on " $start

nohup python trainer/ds2_train.py --data_dir=$data_bucket --vocabulary_file=$voc_file --model_dir=$model_dir --seed=1 >$log_file 2>&1 & 
