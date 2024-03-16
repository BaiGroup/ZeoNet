#!/usr/bin/env bash

EXP=${1:-"001"}
EPOCHS=${2:-30}
BATCHSIZE=${3:-16}
LR=${4:-0.001}
OPTIMIZER=${5:-"Adam"}
MODEL=${6:-"resnet"}
MODELHP=${7:-18}
DSETROOT=${8:-"."}
GRIDRESOLUTION=${9:-0.45}
GRIDSIZE=${10:-100}

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"          
python ./train.py \
    --epochs $EPOCHS \
    --batch_size $BATCHSIZE \
    --lr $LR \
    --optimizer ${OPTIMIZER} \
    --save_dir checkpoints/${EXP}/ \
    --model ${MODEL} \
    --model_hp $MODELHP \
    --dset_root ${DSETROOT} \
    --grid_resolution $GRIDRESOLUTION \
    --grid_size $GRIDSIZE \
