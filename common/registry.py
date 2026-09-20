"""模型与数据集的配置注册表。

所有随「模型」或「数据集」变化的常量都集中到这里，训练/测试脚本仅通过
``--model`` 与 ``--dataset`` 两个参数选择配置，实现代码与具体模型/数据的解耦。
新增模型或数据集时，只需在此追加一条配置，无需改动核心训练/推理代码。
"""
from dataclasses import dataclass


# =============================================================================
# 模型配置
# =============================================================================
@dataclass
class ModelConfig:
    name: str
    # 训练：HuggingFace repo id 或本地路径
    train_path: str
    # 训练时的模型加载器："vision2seq" -> AutoModelForVision2Seq；"causal_lm" -> AutoModelForCausalLM
    loader: str = "vision2seq"
    train_dtype: str = "float16"           # 训练权重加载精度
    image_shortest_edge: int | None = None  # LLaVA-1.5 需把图像 resize 到 336
    # LASER 打分前的答案 token 定位策略：LLaVA 用字符 offset；Qwen/OneVision 搜索 token id
    train_token_locator: str = "find_word"  # "offset" / "find_word"

    # 测试：推理端模型封装类型，决定输入构造/加载类/文本切分标记
    eval_family: str = "qwen"              # "llava" / "qwen" / "onevision"
    # 测试默认权重路径（一般为合并 LoRA 后的模型；权重不随本项目分发，运行时用 --model_path 指定）
    eval_path: str = ""

    def torch_dtype(self):
        import torch

        return {"float16": torch.float16, "bfloat16": torch.bfloat16}[self.train_dtype]


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llava15_7b": ModelConfig(
        name="llava15_7b",
        train_path="llava-hf/llava-1.5-7b-hf",
        loader="vision2seq",
        train_dtype="bfloat16",
        image_shortest_edge=336,
        train_token_locator="offset",
        eval_family="llava",
    ),
    "llava15_13b": ModelConfig(
        name="llava15_13b",
        train_path="llava-hf/llava-1.5-13b-hf",
        loader="vision2seq",
        train_dtype="bfloat16",
        image_shortest_edge=336,
        train_token_locator="offset",
        eval_family="llava",
    ),
    "qwen25_vl_7b": ModelConfig(
        name="qwen25_vl_7b",
        train_path="Qwen/Qwen2.5-VL-7B-Instruct",
        loader="vision2seq",
        train_dtype="float16",
        eval_family="qwen",
    ),
    "oneVision15_8b": ModelConfig(
        name="oneVision15_8b",
        train_path="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        loader="causal_lm",
        train_dtype="float16",
        eval_family="onevision",
    ),
}


# =============================================================================
# 数据集配置
# =============================================================================
@dataclass
class DatasetConfig:
    name: str
    task_type: str                 # "caption"（COCO 幻觉）或 "vqa"（CLEVR / SpatialMQA）
    train_path: str = ""
    test_path: str = ""
    select: int | None = None      # 训练时随机抽取的样本数（None 表示全量）
    # LASER 正则项超参
    beta_sr_train: float = 0.1     # 训练端 β_sr
    beta_sr_eval: float = 0.1      # 测试端 β_sr
    ref_logp_train: float = -28.0  # 训练端参考 logp 常量
    # VQA 专用：候选答案选择方式 "fixed_option"（Spatial）/ "per_index"（CLEVR）
    vqa_candidate_mode: str | None = None
    eval_metric: str = "auroc"     # 测试主指标风格："caption"(含 FPR95/ECE) 或 "vqa"(含 ACC)


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "coco": DatasetConfig(
        name="coco",
        task_type="caption",
        train_path="./data/coco2014_llava_grpo_train",
        test_path="./data/coco_ground_truth.json",
        select=4000,
        beta_sr_train=0.05,
        beta_sr_eval=0.1,
        eval_metric="caption",
    ),
    "spatial": DatasetConfig(
        name="spatial",
        task_type="vqa",
        train_path="./data/spatialmqa_llava_grpo_train",
        test_path="./data/spatialmqa_llava_grpo_test",
        select=2000,
        beta_sr_train=0.1,
        beta_sr_eval=0.1,
        vqa_candidate_mode="fixed_option",
        eval_metric="vqa",
    ),
    "clevr": DatasetConfig(
        name="clevr",
        task_type="vqa",
        train_path="./data/clevr_llava_grpo_train",
        test_path="./data/clevr_llava_grpo_train_test",
        select=None,
        beta_sr_train=0.1,
        beta_sr_eval=0.1,
        vqa_candidate_mode="per_index",
        eval_metric="vqa",
    ),
}


def get_model_config(name: str) -> ModelConfig:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def get_dataset_config(name: str) -> DatasetConfig:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]


def eval_ref_logp(model_cfg: ModelConfig, dataset_cfg: DatasetConfig) -> float:
    """测试端参考 logp 常量：LLaVA 在 caption 任务用 -26，其余情形用 -28。"""
    if model_cfg.eval_family == "llava" and dataset_cfg.task_type == "caption":
        return -26.0
    return -28.0
