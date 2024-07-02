#!/bin/bash


MODELS=( "resnet50_gn_timm" "vitbase_timm" )


EXP_TYPE="severity_shift"
LEVELS=( 5 )
LR_FACTORS=( 1 )

# METHODS=( "no_adapt" "poem" "eata" "sar" "tent" "cotta" )
METHODS=( "cotta" )

S=2024

for METHOD in ${METHODS[@]}; do
    if [ "$METHOD" == "no_adapt" ]; then
        # Run only one seed for no_adapt method
        N_EXPS=10
    else
        # Run multiple seeds for other methods
        N_EXPS=10
    fi

    for (( i=S; i<N_EXPS+S; i++ )); do
        echo seed $i
        for LEVEL in ${LEVELS[@]}; do
            for MODEL in ${MODELS[@]}; do
                bash ./sbatch_job.sh "cd .. ; python3 main.py --seed $i --exp_type $EXP_TYPE --model $MODEL --method $METHOD --level $LEVEL --severity_list 1 2 3 4 5 4 3 2 1 --cont_size 1000 --test_batch_size 1"
            done
        done
    done
done


for METHOD in ${METHODS[@]}; do
    if [ "$METHOD" == "no_adapt" ]; then
        # Run only one seed for no_adapt method
        N_EXPS=10
    else
        # Run multiple seeds for other methods
        N_EXPS=10
    fi

    for (( i=S; i<N_EXPS+S; i++ )); do
        echo seed $i
        for LEVEL in ${LEVELS[@]}; do
            for MODEL in ${MODELS[@]}; do
                bash ./sbatch_job.sh "cd .. ; python3 main.py --seed $i --exp_type $EXP_TYPE --model $MODEL --method $METHOD --level $LEVEL --severity_list 5 4 3 2 1 2 3 4 5 --cont_size 1000 --test_batch_size 1"
            done
        done
    done
done

