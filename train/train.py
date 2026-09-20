"""TruthLens 统一训练入口（GRPO + LASER）。

用法示例：
    python train/train.py --model qwen25_vl_7b --dataset spatial \
        --output_dir ./output/qwen25_vl_7b_spatial

模型与数据集的差异全部由 ``common/registry.py`` 配置，脚本本身与具体模型/数据无关。
"""
import argparse
import os
import sys

# 保证可从仓库根目录导入 common / train / utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("WANDB_DISABLED", "true")

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

from common.registry import get_dataset_config, get_model_config
from train.grpo_config import GRPOConfig
from train.grpo_trainer import GRPOTrainer
from train.laser_tasks import build_laser_task

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHAIR_PKL = os.path.join(_ROOT, "utils", "chair.pkl")
DEFAULT_DEEPSPEED = os.path.join(_ROOT, "configs", "zero2.json")


def parse_args():
    p = argparse.ArgumentParser(description="TruthLens GRPO+LASER training")
    p.add_argument("--model", required=True, help="模型名，见 MODEL_REGISTRY")
    p.add_argument("--dataset", required=True, help="数据集名，见 DATASET_REGISTRY")
    p.add_argument("--model_path", default=None, help="覆盖模型权重路径/repo id")
    p.add_argument("--dataset_path", default=None, help="覆盖训练数据集路径")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--chair_pkl", default=DEFAULT_CHAIR_PKL, help="caption 任务的 CHAIR 缓存")
    p.add_argument("--deepspeed", default=DEFAULT_DEEPSPEED, help="DeepSpeed 配置 json 路径（默认 configs/zero2.json，传空串可禁用）")
    p.add_argument("--select", type=int, default=None, help="覆盖训练样本抽样数")
    # 训练超参
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--loss_type", default="grpo")
    # DeepSpeed launcher 会自动注入 --local_rank=N；同时兼容 torchrun 的 --local-rank。
    p.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    return p.parse_args()


def build_model_and_processor(model_cfg, model_path):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    loader = AutoModelForVision2Seq if model_cfg.loader == "vision2seq" else AutoModelForCausalLM
    model = loader.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=model_cfg.torch_dtype()
    )
    # LLaVA-1.5 需要把图像分辨率固定到 336
    if model_cfg.image_shortest_edge is not None and hasattr(processor, "image_processor"):
        ip = processor.image_processor
        if hasattr(ip, "size"):
            ip.size = {"shortest_edge": model_cfg.image_shortest_edge}
        if hasattr(ip, "crop_size"):
            ip.crop_size = {"height": model_cfg.image_shortest_edge, "width": model_cfg.image_shortest_edge}
    return model, processor


def build_dataset(dataset_path, select, seed):
    ds = load_from_disk(dataset_path)
    if select is not None:
        ds = ds.shuffle(seed=seed).select(range(select))

    def normalize_messages(example):
        msg = example["prompt"]
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            example["prompt"] = [{"role": r, "content": c} for r, c in zip(msg["role"], msg["content"])]
        return example

    def keep_until_last_user(example):
        msgs = example["prompt"]
        last_user = max(i for i, m in enumerate(msgs) if m["role"] in ["user", "human"])
        example["prompt"] = msgs[: last_user + 1]
        return example

    return ds.map(normalize_messages).map(keep_until_last_user)


def main():
    args = parse_args()
    model_cfg = get_model_config(args.model)
    dataset_cfg = get_dataset_config(args.dataset)

    model_path = args.model_path or model_cfg.train_path
    dataset_path = args.dataset_path or dataset_cfg.train_path
    select = args.select if args.select is not None else dataset_cfg.select

    model, processor = build_model_and_processor(model_cfg, model_path)
    ds = build_dataset(dataset_path, select, args.seed)
    laser_task = build_laser_task(model_cfg, dataset_cfg, args.chair_pkl)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # reward 恒为 0：训练信号完全来自 LASER 正则项
    def reward_zero(completions, **kwargs):
        return [0.0 for _ in completions]

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        deepspeed=args.deepspeed or None,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
        loss_type=args.loss_type,
        beta=args.beta,
        seed=args.seed,
        local_rank=args.local_rank,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=processor,
        peft_config=lora_config,
        reward_funcs=[reward_zero],
        laser_task=laser_task,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
