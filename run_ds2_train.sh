#!/bin/bash
# Script to run _locally_  the Deep Speech 2 model training with Common Voice corpus (Mozilla)
# Edit the trainer/config.json to set main parameters for:
# --data_dir
# --vocabulary_file
# --model_dir
# etc.

# Log file
log_file=log_`date +%Y-%m-%d`

start=`date +%s`
echo "Model training and evaluation started on " $start

nohup python trainer/ds2_train.py  --num_gpus=0 --seed=1 >$log_file 2>&1 &

#end=`date +%s`
#runtime=$((end-start))
#echo "Model training time is" $runtime "seconds."
