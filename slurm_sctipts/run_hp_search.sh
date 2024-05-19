#!/bin/bash

EPS_CLIP_VALS=( 0.90 )
GAMMA_VALS=( 0.036 )

MODELS=( "resnet50_gn_timm" "vitbase_timm" )
LOSS_FNS=( "mse" )

EXP_TYPES=( "natural_shift" )
LEVELS=( 5 )
LR_FACTORS=( 1 )

METHODS=( "no_adapt" "poem" "eata" "sar" "tent" )
S=2024

for METHOD in ${METHODS[@]}; do
    if [ "$METHOD" == "no_adapt" ]; then
        # Run only one seed for no_adapt method
        N_EXPS=1
    else
        # Run multiple seeds for other methods
        N_EXPS=10
    fi

    for (( i=S; i<N_EXPS+S; i++ )); do
        echo seed $i
        for GAMMA in ${GAMMA_VALS[@]}; do
            for EPS_CLIP_VAL in ${EPS_CLIP_VALS[@]}; do
                for LR_FACTOR in ${LR_FACTORS[@]}; do
                    for EXP_TYPE in ${EXP_TYPES[@]}; do
                        for LOSS_FN in ${LOSS_FNS[@]}; do
                            for LEVEL in ${LEVELS[@]}; do
                                for MODEL in ${MODELS[@]}; do
                                    bash ./sbatch_job.sh "python3 main.py --seed $i --exp_type $EXP_TYPE --model $MODEL --method $METHOD --level $LEVEL --loss_fn $LOSS_FN --lr_factor $LR_FACTOR --gamma $GAMMA --eps_clip $EPS_CLIP_VAL --test_batch_size 1 --cont_size 5000 --exp_comment new_clipping"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done


