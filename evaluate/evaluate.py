"""TruthLens 统一测试入口。

用法示例：
    python evaluate/evaluate.py --model qwen25_vl_7b --dataset spatial \
        --model_path /path/to/merged_model

caption（COCO）与 vqa（CLEVR/SpatialMQA）两类评测共用同一套 LVLM 与指标模块，
仅数据读取循环不同。模型权重不随本项目分发，请用 --model_path 指定合并后的权重。
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets import load_from_disk

from common.registry import get_dataset_config, get_model_config
from evaluate.lvlm import LVLM
from evaluate.metrics import summarize

DEFAULT_CHAIR_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "chair.pkl")
CAPTION_THRESHOLDS = [0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.3]


def parse_args():
    p = argparse.ArgumentParser(description="TruthLens evaluation")
    p.add_argument("--model", required=True, help="模型名，见 MODEL_REGISTRY")
    p.add_argument("--dataset", required=True, help="数据集名，见 DATASET_REGISTRY")
    p.add_argument("--model_path", required=True, help="推理权重路径（合并 LoRA 后的模型）")
    p.add_argument("--dataset_path", default=None, help="覆盖测试数据集路径")
    p.add_argument("--coco_image_dir", default="./data/coco/val2014")
    p.add_argument("--chair_pkl", default=DEFAULT_CHAIR_PKL)
    p.add_argument("--num_data", type=int, default=5000)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--inference_temp", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_caption(lvlm, args, dataset_cfg):
    ann_path = args.dataset_path or dataset_cfg.test_path
    with open(ann_path, "r") as f:
        coco_data = [json.loads(line) for line in f]
    coco_gt = random.sample(coco_data, min(args.num_data, len(coco_data)))
    question = "Describe the given image in detail."

    true_scores, false_scores = [], []
    for count, entry in enumerate(tqdm(coco_gt, desc="Processing Images"), 1):
        image_path = os.path.join(args.coco_image_dir, entry["image"])
        if not os.path.exists(image_path):
            continue
        image = Image.open(image_path).convert("RGB")
        _, ts, fs = lvlm.generate(image, question, args, img_id=entry["image_id"])
        true_scores.extend(ts)
        false_scores.extend(fs)
        if count % 50 == 0 and true_scores and false_scores:
            summarize(true_scores, false_scores, metric="caption", thresholds=CAPTION_THRESHOLDS)
    summarize(true_scores, false_scores, metric="caption", thresholds=CAPTION_THRESHOLDS)


def eval_vqa(lvlm, args, dataset_cfg):
    ds_path = args.dataset_path or dataset_cfg.test_path
    ds = load_from_disk(ds_path)
    n = min(args.num_data, len(ds))
    data_list = [ds[i] for i in random.sample(range(len(ds)), n)]

    true_scores, false_scores = [], []
    for count, entry in enumerate(tqdm(data_list, desc="Processing Images"), 1):
        image_path = entry["images"][0]
        if not os.path.exists(image_path):
            continue
        question = entry["prompt"]["content"][0]
        image = Image.open(image_path).convert("RGB")
        _, ts, fs = lvlm.generate(
            image, question, args, answer=entry["answer"], index=entry.get("index"),
        )
        true_scores.extend(ts)
        false_scores.extend(fs)
        if count % 50 == 0 and true_scores and false_scores:
            summarize(true_scores, false_scores, metric="vqa")
    summarize(true_scores, false_scores, metric="vqa")


def main():
    args = parse_args()
    fix_seed(args.seed)
    model_cfg = get_model_config(args.model)
    dataset_cfg = get_dataset_config(args.dataset)

    lvlm = LVLM(model_cfg, dataset_cfg, args.model_path, chair_pkl=args.chair_pkl, device=args.device)

    if dataset_cfg.task_type == "caption":
        eval_caption(lvlm, args, dataset_cfg)
    else:
        eval_vqa(lvlm, args, dataset_cfg)


if __name__ == "__main__":
    main()
