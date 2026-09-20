"""VQA 任务（CLEVR / SpatialMQA）共享的答案匹配工具。

训练与测试两端都会用到这里的候选答案表与词元定位逻辑，集中在此避免重复。
"""
import re


# 各类问题对应的候选答案集合。CLEVR 按问题类型取对应候选，SpatialMQA 固定用 ``option``。
CLEVER_DICT = {
    "exist": ["no", "yes"],
    "query_material": ["metal", "rubber"],
    "equal_shape": ["no", "yes"],
    "count": ["2", "6", "0", "1", "3", "5", "4", "7", "8", "9", "10"],
    "query_color": ["purple", "cyan", "brown", "yellow", "red", "blue", "gray", "green"],
    "equal_color": ["yes", "no"],
    "greater_than": ["yes", "no"],
    "equal_material": ["no", "yes"],
    "query_shape": ["sphere", "cylinder", "cube"],
    "less_than": ["no", "yes"],
    "equal_size": ["no", "yes"],
    "query_size": ["large", "small"],
    "equal_integer": ["no", "yes"],
    "option": ["behind", "below", "in front of", "left of", "on", "right of"],
}


def match_candidate(decoded, candidate_ans):
    """在生成文本中，用完整词边界匹配候选答案，返回命中的候选列表。"""
    decoded = decoded.lower().strip()
    matches = []
    for c in candidate_ans:
        c_norm = c.lower().strip()
        # 用正则保证是完整 token，不会把 "2" 匹配到 "12"
        pattern = r"\b" + re.escape(c_norm) + r"\b"
        if re.search(pattern, decoded):
            matches.append(c)
    return matches


def find_word(detected_word, ids, processor, model="QwenVL25"):
    """尝试若干前后缀/大小写变体，找到 ``detected_word`` 对应且出现在 ``ids`` 中的首个 token id。"""
    prefixs = ["", " "]
    suffixs = ["", "s", "es"]
    for word in (detected_word, detected_word.capitalize()):
        for prefix in prefixs:
            for suffix in suffixs:
                token = None
                if "intern" in model:
                    inputs = processor(prefix + word + suffix, return_tensors="pt")
                    token = inputs["input_ids"][0, 0]
                elif model in ["mPLUG_Owl3", "QwenVL25", "llava-one-version1.5", "Pixtral"]:
                    inputs = processor.encode(prefix + word + suffix)
                    token = inputs[0] if len(inputs) else None
                if token is None:
                    continue
                if token in ids:
                    return token
    return None
