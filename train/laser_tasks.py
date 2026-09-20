"""LASER 正则项（数据集相关部分）。

GRPOTrainer 的策略损失与优化流程对所有数据集通用；真正随数据集变化的只有
「如何在生成文本中定位需要施加约束的 token，并给出其目标值」这一步。此处将该逻辑
抽象为可插拔的 ``LaserTask``：

- ``CaptionLaserTask``  用于 COCO 图像描述，借助 CHAIR 定位召回/幻觉物体词；
- ``VQALaserTask``      用于 CLEVR / SpatialMQA，通过候选答案匹配定位答案词。

每个 task 暴露:
  * ``required_fields``：需要从数据集透传到损失计算的列名；
  * ``compute_mse(trainer, inputs, completion_ids, special_token_logps)``：
        返回 ``(mse_sum, hits, alpha)``，由 trainer 汇总进总损失。
"""
import re

import nltk
import torch

from common.chair_loader import load_chair
from common.vqa_utils import CLEVER_DICT, match_candidate, find_word


def locate_token_positions(locator, word, decoded, completion_ids, tokenizer):
    """按模型家族使用原始代码中的 token 定位方式。

    - LLaVA: 重新 tokenize 生成文本，通过 ``offset_mapping`` 将答案字符跨度映射到首个 token；
    - Qwen/OneVision: 用 ``find_word`` 找 token id，再取它在 completion 中的全部位置。
    """
    if locator == "offset":
        offsets = tokenizer(decoded, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        positions = []
        for match in re.finditer(rf"\b{re.escape(word.lower())}\b", decoded.lower()):
            start_char, end_char = match.span()
            overlapping = [
                i for i, (start, end) in enumerate(offsets)
                if start < end_char and end > start_char
            ]
            if overlapping:
                positions.append(overlapping[0])
        return positions

    if locator == "find_word":
        token = find_word(word, completion_ids, tokenizer, "QwenVL25")
        if token is None:
            return []
        return torch.where(completion_ids == token)[0].tolist()

    raise ValueError(f"Unknown token locator: {locator}")


class LaserTask:
    required_fields: list[str] = []
    alpha: float = 1.0

    def compute_mse(self, trainer, inputs, completion_ids, special_token_logps):
        raise NotImplementedError


class CaptionLaserTask(LaserTask):
    """COCO caption：用 CHAIR 找出召回物体词（目标 1）与幻觉物体词（目标 0）。"""

    required_fields = ["image_id"]

    def __init__(self, chair_pkl_path: str, token_locator: str,
                 beta_sr: float = 0.05, ref_logp: float = -28.0, alpha: float = 1.0):
        self.evaluator = load_chair(chair_pkl_path)
        self.token_locator = token_locator
        self.beta_sr = beta_sr
        self.ref_logp = ref_logp
        self.alpha = alpha

    def compute_mse(self, trainer, inputs, completion_ids, special_token_logps):
        device = special_token_logps.device
        bs = completion_ids.size(0)
        mse_sum = torch.zeros([bs], device=device)
        hits = torch.zeros([bs], device=device)
        tokenizer = trainer.processing_class.tokenizer
        ref = torch.tensor(self.ref_logp, device=device)

        for b in range(bs):
            image_id = int(inputs["image_id"][b])
            decoded = tokenizer.decode(completion_ids[b], skip_special_tokens=True).strip()
            tokens = nltk.word_tokenize(decoded.lower())
            if ("toilet" in tokens) and ("seat" in tokens):
                tokens = [word for word in tokens if word != "seat"]

            cap_dict = self.evaluator.compute_hallucinations(image_id, decoded)
            recall_idxs = cap_dict.get("recall_idxs", [])
            hallucination_idxs = cap_dict.get("hallucination_idxs", [])
            NR, NH = len(recall_idxs), len(hallucination_idxs)
            w_pos = (NR + NH) / (2 * max(NR, 1))
            w_neg = (NR + NH) / (2 * max(NH, 1))

            true_matrix, false_matrix = {}, {}
            for wid, target, weight, seen in (
                [(i, 1.0, w_pos, true_matrix) for i in recall_idxs]
                + [(i, 0.0, w_neg, false_matrix) for i in hallucination_idxs]
            ):
                if wid >= len(tokens):
                    continue
                words = tokens[wid]
                if seen.get(words) is not None:
                    continue
                seen[words] = True
                positions = locate_token_positions(
                    self.token_locator, words, decoded, completion_ids[b], tokenizer
                )
                for idx in positions:
                    if idx >= special_token_logps.size(1):
                        continue
                    diff = self.beta_sr * (special_token_logps[b, idx] - ref.detach())
                    mse_sum[b] += weight * (diff - target) ** 2
                    hits[b] += 1
        return mse_sum, hits, self.alpha


class VQALaserTask(LaserTask):
    """CLEVR / SpatialMQA：匹配预测答案，对答案 token 施加约束（正确目标 1，错误目标 0）。"""

    required_fields = ["answer", "index"]

    def __init__(self, candidate_mode: str, token_locator: str,
                 beta_sr: float = 0.1, ref_logp: float = -28.0, alpha: float = 1.0):
        self.candidate_mode = candidate_mode
        self.token_locator = token_locator
        self.beta_sr = beta_sr
        self.ref_logp = ref_logp
        self.alpha = alpha

    def _candidates(self, inputs, b):
        if self.candidate_mode == "per_index":
            return CLEVER_DICT[inputs["index"][b]]
        return CLEVER_DICT["option"]

    def compute_mse(self, trainer, inputs, completion_ids, special_token_logps):
        device = special_token_logps.device
        bs = completion_ids.size(0)
        mse_sum = torch.zeros([bs], device=device)
        hits = torch.zeros([bs], device=device)
        tokenizer = trainer.processing_class.tokenizer
        ref = torch.tensor(self.ref_logp, device=device)

        # 第一遍：统计正确/错误样本数，用于正负样本平衡权重
        decoded_list, cand_list = [], []
        NR = NH = 0
        for b in range(bs):
            candidate = self._candidates(inputs, b)
            ans = inputs["answer"][b].lower().strip()
            decoded = tokenizer.decode(completion_ids[b], skip_special_tokens=True)
            decoded_list.append(decoded)
            cand_list.append(candidate)
            match = match_candidate(decoded, candidate)
            if not match:
                continue
            if match[0].lower().strip() == ans:
                NR += 1
            else:
                NH += 1
        w_pos = (NR + NH) / (2 * max(NR, 1))
        w_neg = (NR + NH) / (2 * max(NH, 1))

        # 第二遍：定位答案词元并累加 MSE
        for b in range(bs):
            ans = inputs["answer"][b].lower().strip()
            match = match_candidate(decoded_list[b], cand_list[b])
            if not match:
                continue
            pred = match[0].lower().strip()
            correct = pred == ans
            positions = locate_token_positions(
                self.token_locator, pred, decoded_list[b], completion_ids[b], tokenizer
            )
            target = 1.0 if correct else 0.0
            weight = w_pos if correct else w_neg
            for idx in positions:
                if idx >= special_token_logps.size(1):
                    continue
                diff = self.beta_sr * (special_token_logps[b, idx] - ref.detach())
                mse_sum[b] += weight * (diff - target) ** 2
                hits[b] += 1
        return mse_sum, hits, self.alpha


def build_laser_task(model_cfg, dataset_cfg, chair_pkl_path: str) -> LaserTask:
    """根据模型家族与数据集配置构建 LASER task。"""
    if dataset_cfg.task_type == "caption":
        return CaptionLaserTask(
            chair_pkl_path=chair_pkl_path,
            token_locator=model_cfg.train_token_locator,
            beta_sr=dataset_cfg.beta_sr_train,
            ref_logp=dataset_cfg.ref_logp_train,
        )
    elif dataset_cfg.task_type == "vqa":
        return VQALaserTask(
            candidate_mode=dataset_cfg.vqa_candidate_mode or "fixed_option",
            token_locator=model_cfg.train_token_locator,
            beta_sr=dataset_cfg.beta_sr_train,
            ref_logp=dataset_cfg.ref_logp_train,
        )
    raise ValueError(f"Unknown task_type: {dataset_cfg.task_type}")
