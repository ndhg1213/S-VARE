# S-VARE

基于 [Infinity](https://github.com/FoundationVision/Infinity)（BSQ-VAE + 视觉自回归 Transformer）的概念擦除（Concept Erasure）训练框架。


预训练权重默认放在 `../../Models/`：

- `Infinity/infinity_2b_reg.pth`
- `Infinity/infinity_vae_d32_reg.pth`
- `flan-t5-xl/`

## 数据

一份 JSON，每条样本是一对 prompt：

```json
[
  {"idx": 0,
   "original_prompt": "a photo of a man hitting another man",
   "modified_prompt":  "a photo of a man talking with another man"}
]
```

如果原始 JSON 里是 `unsafe_prompt / safe_prompt` 或缺 `idx`，先跑：

```bash
python tools/add_idx_to_json.py -j inputs/raw.json -s inputs/train_xxx.json
```

## 流程

```bash
# 1. 用原模型预存 target / source 两路教师特征 (gt_logits / x_BLC / bit labels)
bash scripts/preprocess_data.sh

# 2. 擦除训练，输出在 local_run/<exp_name>/
bash scripts/train_svare.sh

# 3. 用训练后的权重出图
bash scripts/infer_svare.sh
```

预处理产物：

```
inputs/<name>_data/
├── target/{image,label}/<idx>.{png,pt}   # modified_prompt -> 保持分支
└── source/{image,label}/<idx>.{png,pt}   # original_prompt -> 擦除分支
```

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{zhong2025closing,
  title={Closing the safety gap: Surgical concept erasure in visual autoregressive models},
  author={Zhong, Xinhao and Zhou, Yimin and Zhang, Zhiqi and Li, Junhao and Sun, Yi and Chen, Bin and Xia, Shu-Tao and Wang, Xuan and Xu, Ke},
  journal={arXiv preprint arXiv:2509.22400},
  year={2025}
}
```

