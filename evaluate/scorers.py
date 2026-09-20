"""测试端 LASER 分数计算。

给定生成序列上「特殊 token」的逐位置 logp（``logps``），根据任务在关键词位置计算
检测分数 ``diff = beta_sr * (logp - ref_logp)``，正样本词进入 true_scores，负样本进入 false_scores。
"""
import re

import nltk
import torch

from common.vqa_utils import match_candidate, find_word


def caption_scorer(evaluator, img_id, final_ans, gen_token_ids, logps, tokenizer,
                   beta_sr, ref_logp, mode="find_word"):
    """COCO caption：用 CHAIR 找召回/幻觉物体词，定位其首个 token 并打分。

    mode='offset'   : 通过字符 offset 映射定位 token（LLaVA 使用）；
    mode='find_word': 通过词元变体搜索 token id（Qwen / OneVision 使用）。
    """
    device = logps.device
    ref = torch.tensor(ref_logp, device=device)

    tokens = nltk.word_tokenize(final_ans.lower())
    if ("toilet" in tokens) and ("seat" in tokens):
        tokens = [w for w in tokens if w != "seat"]
    cap_dict = evaluator.compute_hallucinations(img_id, final_ans)
    recall_idxs = cap_dict.get("recall_idxs", [])
    hallucination_idxs = cap_dict.get("hallucination_idxs", [])

    offsets = None
    if mode == "offset":
        enc = tokenizer(final_ans, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]

    true_scores, false_scores = [], []
    for idxs, bucket in ((recall_idxs, true_scores), (hallucination_idxs, false_scores)):
        seen = {}
        for wid in idxs:
            if wid >= len(tokens):
                continue
            words = tokens[wid]
            if seen.get(words) is not None:
                continue
            seen[words] = True
            if mode == "offset":
                pattern = rf"\b{re.escape(words.lower())}\b"
                for m in re.finditer(pattern, final_ans.lower()):
                    start_char, end_char = m.span()
                    token_positions = [
                        i for i, (s, e) in enumerate(offsets) if s < end_char and e > start_char
                    ]
                    if not token_positions:
                        continue
                    e_j = token_positions[0]
                    bucket.append((beta_sr * (logps[0, e_j] - ref)).cpu().numpy())
                    break
            else:
                token = find_word(words, gen_token_ids, tokenizer, "QwenVL25")
                if token is not None and torch.where(gen_token_ids == token)[0].numel() != 0:
                    toke_idx = torch.where(gen_token_ids == token)[0][0]
                    bucket.append((beta_sr * (logps[0, toke_idx] - ref)).cpu().numpy())
    return true_scores, false_scores


def vqa_scorer(final_ans, answer, candidate, gen_token_ids, logps, tokenizer, beta_sr, ref_logp):
    """CLEVR / SpatialMQA：匹配预测答案，对答案词首个 token 打分。"""
    device = logps.device
    ref = torch.tensor(ref_logp, device=device)

    match = match_candidate(final_ans, candidate)
    if len(match) == 0:
        return [], []
    pred = match[0].lower().strip()
    ans = answer.lower().strip()
    correct = pred == ans

    true_scores, false_scores = [], []
    token = find_word(pred, gen_token_ids, tokenizer, "QwenVL25")
    if token is not None and torch.where(gen_token_ids == token)[0].numel() != 0:
        toke_idx = torch.where(gen_token_ids == token)[0][0]
        diff = (beta_sr * (logps[0, toke_idx] - ref)).cpu().numpy()
        (true_scores if correct else false_scores).append(diff)
    return true_scores, false_scores
