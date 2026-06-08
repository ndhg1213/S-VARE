import os
import random
import sys
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch
from tap import Tap

import infinity.utils.dist as dist


class Args(Tap):
    # =====================================================================
    # ===================== Concept-erasure data / loss ===================
    # =====================================================================
    data_path: str = ''                 # training prompts json
    gt_info_path: str = ''              # pre-computed teacher (preserve) features dir
    neg_info_path: str = ''             # pre-computed teacher (target) features dir
    finetuning_batch_size: int = 2
    with_reg: int = 1                   # 0: no prior distillation, 1: prior distillation
    reg_lambda: float = 0.5             # regularization strength
    train_iter: int = 500
    val_prompts: str = ''               # validation prompts separated by ';'

    # =====================================================================
    # =========================== Selective tuning ========================
    # =====================================================================
    enable_sa_layer: int = 0
    enable_ca_layer: int = 1
    enable_ffn_layer: int = 1
    enable_all_layer: int = 0
    enable_lora: int = 0
    lora_rank: int = 4
    rwe: bool = False                   # randomly init word emb and freeze it

    # =====================================================================
    # ============================ Model & ckpts ==========================
    # =====================================================================
    model_type: str = 'infinity_2b'
    model: str = '2bc8'                 # gpt model alias for infinity (e.g. 2bc8 -> infinity_2bc8)
    model_alias: str = 'b'              # [auto-set]
    model_path: str = '../../Models/Infinity/infinity_2b_reg.pth'
    vae_type: int = 32                  # bsq vae quant bits (16/32/64)
    vae_path: str = '../../Models/Infinity/infinity_vae_d32_reg.pth'
    t5_path: str = '../../Models/flan-t5-xl'
    online_t5: bool = True
    tlen: int = 512                     # text encoder max length
    text_prefix_length: int = 4         # length of S* text embedding
    text_channels: int = 2048
    pn: str = '1M'                      # pixel nums, 0.06M / 0.25M / 1M
    h_div_w_template: float = 1.000
    apply_spatial_patchify: int = 0
    use_flex_attn: bool = False
    add_lvl_embeding_only_first_block: int = 1
    use_bit_label: int = 1
    rope2d_each_sa_layer: int = 1
    rope2d_normalized_by_hw: int = 2
    always_training_scales: int = 100   # truncate training scales
    enable_checkpointing: str = 'full-block'  # full-block / full-attn / self-attn / None

    # =====================================================================
    # ========================== Optim / Schedule =========================
    # =====================================================================
    fp16: int = 2                       # 1: fp16, 2: bf16
    bf16: int = 1
    tf32: bool = True
    sdpa_mem: bool = True
    tfast: int = 0                      # torch.compile mode
    afuse: bool = True                  # fused adam
    nowd: int = 1                       # disable weight decay on sparse params
    ada: str = '0.9_0.97'               # adam betas
    opt: str = 'adamw'
    oeps: float = 0
    tblr: float = 6e-3
    tlr: float = None                   # [auto-set] = ac * tblr * glb_batch_size / 256
    twd: float = 0.005
    twde: float = 0
    tclip: float = 5.                   # >100 for per-param clip (%= 100)
    sche: str = 'lin0'                  # cos / exp / lin0 / ...
    wp: float = 0.00000001
    wp0: float = 0.005
    wpe: float = 1.0
    cdec: bool = False                  # decay grad clip thresholds
    ep: int = 1000

    # batch / accumulation
    lbs: int = 4                        # local batch size (overrides bs)
    bs: int = 0                         # global batch size
    batch_size: int = 0                 # [auto-set] per-GPU
    glb_batch_size: int = 0             # [auto-set]
    ac: int = 1                         # gradient accumulation
    r_accu: float = 1.0                 # [auto-set] = 1 / ac
    workers: int = 8

    # =====================================================================
    # ============================ Sampling ===============================
    # =====================================================================
    cfg: float = 4
    tau: float = 0.5                    # softmax temperature in sampling
    cfg_insertion_layer: int = 0
    sampling_per_bits: int = 1

    # =====================================================================
    # ============================ Logging / IO ===========================
    # =====================================================================
    local_out_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')
    bed: str = ''
    exp_name: str = ''
    project_name: str = 'Infinity'
    log_freq: int = 50
    log_every_iter: bool = False
    save_txt_iters_freq: int = 100
    save_model_iters_freq: int = 100
    prof_freq: int = 50
    wandb: int = 0
    log_txt_path: str = ''              # [auto-set]

    # =====================================================================
    # ============================== Misc =================================
    # =====================================================================
    seed: int = None
    rand: bool = True                   # actual seed = seed + (dist.get_rank()*512 if rand else 0)
    device: str = 'cpu'                 # [auto-set]
    auto_resume: bool = True
    resume_ckpts: str = ''
    use_fsdp_model_ema: int = 0
    stable: bool = False
    gpt_training: bool = True           # [auto-set via property below]

    # debug-only
    dbg: bool = 'KEVIN_LOCAL' in os.environ

    @property
    def is_gpt_training(self) -> bool:
        return len(self.model) > 0

    def set_initial_seed(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            seed = self.seed + (dist.get_rank() * 512 if self.rand else 0)
            torch.backends.cudnn.deterministic = True
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed + dist.get_rank() * 512)
        return g

    def compile_model(self, m, fast):
        if fast == 0:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m

    def state_dict(self, key_ordered: bool = True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k != 'device':
                d[k] = getattr(self, k)
        return d

    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):
            d = eval('\n'.join(l for l in d.splitlines() if '<bound' not in l and 'device(' not in l))
        for k in d.keys():
            if k in {'is_large_model', 'gpt_training'}:
                continue
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e

    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')

    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k != 'device':
                s.append(f'  {k:20s}: {getattr(self, k)}')
        return f'{{\n' + '\n'.join(s) + '\n}}\n'


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)

    if len(args.extra_args) > 0:
        print('======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print('======================================================================================\n\n')

    args.set_tf32(args.tf32)
    if args.dbg:
        torch.autograd.set_detect_anomaly(True)

    try: os.makedirs(args.bed, exist_ok=True)
    except Exception: pass
    try: os.makedirs(args.local_out_path, exist_ok=True)
    except Exception: pass

    day3 = 60 * 24 * 3
    dist.init_distributed_mode(
        local_out_path=args.local_out_path,
        fork=False,
        timeout_minutes=day3 if int(os.environ.get('LONG_DBG', '0') or '0') > 0 else 30,
    )

    args.gpt_training = args.is_gpt_training
    args.device = dist.get_device()
    args.r_accu = 1 / args.ac
    args.rand |= args.seed is None
    args.sche = args.sche or ('lin0' if args.gpt_training else 'cos')
    if args.wp == 0:
        args.wp = args.ep * 1 / 100
    args.ada = args.ada or ('0.9_0.96' if args.gpt_training else '0.5_0.9')
    args.opt = args.opt.lower().strip()

    if args.lbs:
        bs_per_gpu = args.lbs / args.ac
    else:
        bs_per_gpu = args.bs / args.ac / dist.get_world_size()
    bs_per_gpu = 1
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size()
    args.workers = min(args.workers, bs_per_gpu)
    args.tlr = args.ac * args.tblr * args.glb_batch_size / 256
    args.twde = args.twde or args.twd

    if args.gpt_training:
        assert args.vae_path, 'VAE path must be specified when training GPT'
        from infinity.models import alias_dict, alias_dict_inv
        if args.model in alias_dict:
            args.model = alias_dict[args.model]
            args.model_alias = alias_dict_inv[args.model]
        else:
            args.model_alias = args.model
            args.model = f'infinity_{args.model}'

    args.log_txt_path = os.path.join(args.local_out_path, 'log.txt')

    args.enable_checkpointing = None if args.enable_checkpointing in [False, 0, '0'] else args.enable_checkpointing
    args.enable_checkpointing = 'full-block' if args.enable_checkpointing in [True, 1, '1'] else args.enable_checkpointing
    assert args.enable_checkpointing in [None, 'full-block', 'full-attn', 'self-attn'], (
        f'only support no-checkpointing or full-block/full-attn checkpointing, but got {args.enable_checkpointing}.'
    )

    if len(args.exp_name) == 0:
        args.exp_name = os.path.basename(args.bed) or 'test_exp'

    if dist.is_master():
        os.system(f'rm -rf {os.path.join(args.bed, "ready-node*")} {os.path.join(args.local_out_path, "ready-node*")}')

    if args.sdpa_mem:
        from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(False)

    return args
