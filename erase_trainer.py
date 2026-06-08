import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import infinity.utils.dist as dist
from infinity.models import Infinity
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.amp_opt import AmpOptimizer
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


FTen = torch.Tensor
ITen = torch.LongTensor


class InfinityTrainer(object):
    def __init__(
        self,
        vae_local,
        gpt: DDP,
        gpt_opt: AmpOptimizer,
        dbg_unused: bool = False,
    ):
        super().__init__()
        self.gpt = gpt
        self.vae_local = vae_local
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.dbg_unused = dbg_unused
        self.batch_size = 0
        self.seq_len = 0
        self.train_loss = nn.CrossEntropyLoss(reduction='none')

    def train_step(
        self,
        ep: int,
        it: int,
        g_it: int,
        stepping: bool,
        clip_decay_ratio: float,
        metric_lg: misc.MetricLogger,
        logging_params: bool,
        gt_logits_BLV: Union[ITen, FTen],
        x_BLC_wo_prefix: Union[ITen, FTen],
        gt_ms_idx_Bl: Union[ITen, FTen],
        gt_logits_BLVs_wo_inf: Union[ITen, FTen],
        neg_logits_BLV: Union[ITen, FTen],
        neg_BLC_wo_prefix: Union[ITen, FTen],
        neg_ms_idx_Bl: Union[ITen, FTen],
        neg_logits_BLVs_wo_inf: Union[ITen, FTen],
        neg_text_cond_tuple: Union[ITen, FTen],
        pos_text_cond_tuple: Union[ITen, FTen],
        args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:

        device = neg_text_cond_tuple[0].device

        # Build scale schedule for the active aspect ratio (T=1 image case).
        h_div_w = 1.0
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        training_scales = args.always_training_scales

        # =============== 1) Erase loss on negative (to-be-erased) prompt ===============
        with self.gpt_opt.amp_ctx:
            x_BLC_wo_prefix = x_BLC_wo_prefix.to(device)
            gt_ms_idx_Bl = gt_ms_idx_Bl.to(device)

            tg_logits_BLV = self.gpt(
                neg_text_cond_tuple,
                x_BLC_wo_prefix,
                scale_schedule=scale_schedule[:training_scales],
            )
            self.batch_size, self.seq_len = tg_logits_BLV.shape[:2]

            # Per-bit CE loss [B, L, 32]: each token is a 32-bit code, so we predict
            # 32 independent binary logits over {0, 1}.
            bit_loss = self.train_loss(
                tg_logits_BLV.reshape(self.batch_size, self.seq_len, -1, 2).permute(0, 3, 1, 2),
                gt_ms_idx_Bl,
            )

            # Bit-error mask: a bit is "wrong" when its CE loss exceeds the binary
            # decision boundary ln(2) (i.e. predicted prob for the GT class < 0.5).
            # Per-token error rate = fraction of wrong bits in that token's 32-bit code.
            bit_error_threshold = math.log(2.0)
            bit_error = (bit_loss.detach() > bit_error_threshold).float()       # [B, L, 32]
            token_error_rate = bit_error.mean(dim=-1)                           # [B, L]

            # Per-token mean bit loss
            loss = bit_loss.mean(dim=-1)                                        # [B, L]

            # Drop tokens whose within-token error rate is < 25% — they are
            # already well predicted and shouldn't contribute to the optimization.
            token_keep_mask = (token_error_rate >= 0.25).float()                # [B, L]
            loss = loss * token_keep_mask

            lw = 1. / self.seq_len
            loss = loss.mul(lw).sum(dim=-1).mean()

        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(
            ep=ep, it=it, g_it=g_it, stepping=stepping, logging_params=logging_params,
            loss=loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable,
        )

        if stepping:
            self._maybe_dbg_unused()
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)

        # =============== 2) Distillation loss on positive (preserved) prompt ===============
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=False):
                stu_logits_BLV = self.gpt(
                    pos_text_cond_tuple,
                    x_BLC_wo_prefix,
                    scale_schedule=scale_schedule[:training_scales],
                )

            gt_logits_BLVs_wo_inf = gt_logits_BLVs_wo_inf.to(device)
            teacher_probs = torch.softmax(
                gt_logits_BLVs_wo_inf.reshape(self.batch_size, -1, 2), dim=-1,
            )
            student_log_probs = torch.log_softmax(
                stu_logits_BLV.reshape(self.batch_size, -1, 2), dim=-1,
            )
            distill_loss = F.kl_div(
                student_log_probs, teacher_probs, reduction='none',
            ).reshape(self.batch_size, self.seq_len, -1).mean(dim=-1)

            distill_loss = distill_loss.mul(lw).sum(dim=-1).mean() * args.reg_lambda

        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(
            ep=ep, it=it, g_it=g_it, stepping=stepping, logging_params=logging_params,
            loss=distill_loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable,
        )

        if stepping:
            self._maybe_dbg_unused()
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)

        # =============== Metric logging ===============
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            res_loss = self.train_loss(tg_logits_BLV, gt_logits_BLV.to(device).detach()).mean()
            grad_for_log = grad_norm_t.item() if grad_norm_t is not None else 0.0
            metrics = torch.tensor([grad_for_log, res_loss], device=loss.device)
            metrics = metrics.cpu().data.numpy() / dist.get_world_size()
            grad_for_log, res_loss = metrics

            metric_lg.update(tnm=grad_for_log)

            if args.wandb:
                wandb_utils.log({"Overall/grad_norm_t": grad_for_log}, step=g_it)

        return grad_norm_t, scale_log2_t

    def _maybe_dbg_unused(self):
        if not self.dbg_unused:
            return
        unused = [n for n, p in self.gpt.named_parameters() if p.grad is None]
        if unused:
            raise AttributeError(f'unused param: {unused}')
