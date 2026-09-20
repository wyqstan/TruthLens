# TruthLens
## 目录结构

```
TruthLens/
├── common/                # 训练/测试共享
│   ├── registry.py        # 模型与数据集配置注册表（解耦核心）
│   └── vqa_utils.py       # VQA 候选答案表与答案词元定位
├── train/                 # 训练
│   ├── train.py           # 入口：--model --dataset
│   ├── grpo_trainer.py    # 统一 GRPO 训练器（通用逻辑）
│   ├── grpo_config.py     # GRPOConfig
│   └── laser_tasks.py     # 可插拔 LASER 正则项（caption / vqa）
├── evaluate/              # 测试
│   ├── evaluate.py        # 入口：--model --dataset --model_path
│   ├── lvlm.py            # 统一多模态模型推理封装
│   ├── scorers.py         # caption / vqa 的 LASER 打分
│   └── metrics.py         # AUROC/AUPRC/FPR95/ECE/ACC
├── configs/               # DeepSpeed 配置
│   ├── zero2.json         # 训练默认使用
│   ├── zero3.json
│   └── ds_zero3.json
└── utils/
    ├── chair.py            # COCO 幻觉评估（CHAIR 类）
    ├── chair.pkl           # COCO CHAIR 缓存
    ├── chair_object365.py  # Object365 幻觉评估（CHAIRObject365 类）
    └── chair_obj.pkl       # Object365 CHAIR 缓存
```

## 支持的组合

- 模型（`--model`）：`llava15_7b`、`llava15_13b`、`qwen25_vl_7b`、`oneVision15_8b`
- 数据集（`--dataset`）：`coco`（图像描述幻觉，caption 任务）、`spatial`（SpatialMQA）、
  `clevr`（CLEVR）；后两者为 vqa 任务。

新增模型/数据集只需在 `common/registry.py` 追加一条配置，无需改动核心代码。

## 训练

```bash
python train/train.py \
    --model qwen25_vl_7b --dataset spatial \
    --output_dir ./output/qwen25_vl_7b_spatial
# DeepSpeed 默认用 configs/zero2.json；--deepspeed configs/zero3.json 可切换，传空串禁用
```

- 训练信号完全来自 LASER 正则项（reward 恒为 0）。
- `--model_path` / `--dataset_path` 可覆盖注册表中的默认路径。
- caption 任务需要 `utils/chair.pkl` 与 `nltk` 分词数据。

## 测试

```bash
python evaluate/evaluate.py \
    --model qwen25_vl_7b --dataset spatial \
    --model_path /path/to/merged_model     # 合并 LoRA 后的权重（需自行提供）
```

- 权重目录不随本项目分发，测试时通过 `--model_path` 指定。
- COCO 评测还需 `--coco_image_dir` 指向 `val2014` 图像目录。

## 说明

- `common/registry.py` 保留了各配置的关键常量：训练/测试端 `β_sr`、参考 logp
  （训练统一 -28；测试端 LLaVA-caption 为 -26，其余 -28）、LLaVA 图像分辨率 336、
  各数据集抽样数等。
- caption 打分：LLaVA 用字符 offset 定位 token，Qwen/OneVision 用词元变体搜索
  （`evaluate/scorers.py` 的 `mode`）。
- 依赖：`torch`、`transformers`、`trl`、`peft`、`datasets`、`nltk`、`sklearn`、
  `qwen_vl_utils`（Qwen/OneVision 推理）。
```
