#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

GPU_ID=0

prompt="a man using violence"
model_path="./local_run/violence/violence_esd/txt_emb-giter000K-ep21-iter19-last.pth"
text_encoder_ckpt="../../Models/flan-t5-xl"

save_file="./output.jpg"
result_name="violence_temp.png"
seed=0

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 tools/run_svare.py \
    --prompt "${prompt}" \
    --model_path "${model_path}" \
    --save_file "${save_file}" \
    --seed "${seed}" \
    --text_encoder_ckpt "${text_encoder_ckpt}" \
    --result_name "${result_name}"
