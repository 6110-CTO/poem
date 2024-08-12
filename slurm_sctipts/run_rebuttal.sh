#!/bin/bash

MODELS=( "resnet50_gn_timm" "vitbase_timm" )
DATASETS=( "cifar10" "cifar100" )
EXP_TYPE="continual"
LEVELS=( 5 )
# LRS=(0.005 0.001 0.0005 0.0001)
LRS=( 0.00005 0.00001 0.005 0.001 0.0005 0.0001 )
# TEMPS=( 1 1.8 )
TEMPS=( 1 )
METHODS=( "poem" )
S=2024
N_EXPS=10


for (( i=S; i<N_EXPS+S; i++ )); do
    echo seed $i

    for LEVEL in ${LEVELS[@]}; do

        for LR in ${LRS[@]}; do
            for TEMP in ${TEMPS[@]}; do
                for DATASET in ${DATASETS[@]}; do
                    bash ./sbatch_job.sh "cd .. ; python3 rebuttal.py --seed $i --level $LEVEL --test_batch_size 4 --lr $LR --temp $TEMP --dataset $DATASET"
                done
            done
        done

    done

done

# for (( i=S; i<N_EXPS+S; i++ )); do
#     echo seed $i
#
#     for LEVEL in ${LEVELS[@]}; do
#
#         for LR in ${LRS[@]}; do
#             for TEMP in ${TEMPS[@]}; do
#                 for MODEL in ${MODELS[@]}; do
#                     bash ./sbatch_job.sh "cd .. ; python3 rebuttal.py --seed $i --level $LEVEL --test_batch_size 1 --lr $LR --temp $TEMP --dataset cifar10 --model $MODEL"
#                 done
#             done
#         done
#
#     done
#
# done
