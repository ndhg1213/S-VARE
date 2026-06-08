#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

GPU_ID=3
LOCAL_OUT=local_run
exp_name=violence/violence_esd
t5_path=../../Models/flan-t5-xl
infinity_model_path=../../Models/Infinity/infinity_2b_reg.pth
mkdir -p $LOCAL_OUT
data_path="./inputs/train_violence.json"
vae_type=32
vae_path=../../Models/Infinity/infinity_vae_d32_reg.pth
finetuning_batch_size=2
gt_info_path=./inputs/train_violence_data/target/label
neg_info_path=./inputs/train_violence_data/source/label
val_prompts="a photo of a man fight with others; a photo of a woman fight with others"
local_out_path=$LOCAL_OUT/${exp_name}

CUDA_VISIBLE_DEVICES=${GPU_ID} python erase_train.py \
    --val_prompts="${val_prompts}" \
    --local_out_path ${local_out_path} \
    --data_path=${data_path} \
    --exp_name=${exp_name} \
    --t5_path=${t5_path} \
    --finetuning_batch_size=${finetuning_batch_size} \
    --seed 42 \
    --gt_info_path=${gt_info_path} \
    --neg_info_path=${neg_info_path}
