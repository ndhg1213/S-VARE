#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=3 python ./tools/preprocess_data.py \
  --json_file ./inputs/train_violence.json \
  --save_path ./inputs/train_violence_data \
  --vae_type 32 \
  --seed 42 \
  --text_encoder_ckpt ../../Models/flan-t5-xl \
  --vae_path ../../Models/Infinity/infinity_vae_d32_reg.pth \
  --model_path ../../Models/Infinity/infinity_2b_reg.pth \
  --pn 1M 