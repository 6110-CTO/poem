#!/bin/bash

MODELS=( "resnet50_gn_timm" "vitbase_timm" )

EXP_TYPE="martingale"
LEVELS=( 1 5 )

METHODS=( "poem" )
S=2024
N_EXPS=1

for METHOD in ${METHODS[@]}; do
    for (( i=S; i<N_EXPS+S; i++ )); do
        echo seed $i
        for LEVEL in ${LEVELS[@]}; do
            for MODEL in ${MODELS[@]}; do
                bash ./sbatch_job.sh "cd .. ; python3 main.py --seed $i --exp_type $EXP_TYPE --model $MODEL --method $METHOD --level $LEVEL --test_batch_size 1"
            done
        done
    done
done
