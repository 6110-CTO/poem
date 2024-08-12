#!/bin/bash

MODELS=( "resnet50_gn_timm" "vitbase_timm" )


S=2024
N_EXPS=10

for (( i=S; i<N_EXPS+S; i++ )); do
    echo seed $i
    for MODEL in ${MODELS[@]}; do
        bash ./sbatch_job.sh "cd .. ; python3  train_office_home_model.py --seed $i --dataset office_home --model $MODEL"
    done
done
