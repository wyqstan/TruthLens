import warnings
from dataclasses import dataclass, field

from transformers import TrainingArguments
@dataclass
class GRPOConfig(TrainingArguments):
    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )

    """
    在反向传播（backward pass）时，PyTorch 默认会保存前向传播（forward pass）中所有中间激活值（activations），
    以便计算梯度。但这些激活值会占用大量显存，尤其在大模型或长序列（如长文本、高分辨率图像）中。
    """
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )

    # 启用 bfloat16 混合精度训练/推理，节省显存并可能加速
    bf16: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )
    # Transformers 4.57.0 introduced a bug that caused the dtype of `lr_scheduler_kwargs` to be unparsable. This issue
    # was fixed in https://github.com/huggingface/transformers/pull/41322, but the fix has not yet been released. We
    # add a temporary workaround here, which can be removed once the fix is available—likely in Transformers 4.57.2.
    # 这个参数用于向学习率调度器（learning rate scheduler）传入额外的超参数，当默认调度器配置不足以满足需求时使用。
    lr_scheduler_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard "
            "restarts."
        },
    )

    # Parameters that control the model and reference model
    model_init_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )
    
    # 是否关掉模型中的dropout层，以确保在使用参考模型时，模型对相同输入不会生成不同的对数概率（logprobs）。
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )

    # 将语言模型的输出头（LM Head）转为 float32 精度，以提高数值稳定性。
    cast_lm_head_to_fp32: bool = field(
        default=False,
        metadata={
            "help": "Whether to cast the language modeling head of the policy and reference, models to float32."
            "As recommended by the [ScaleRL](https://huggingface.co/papers/2510.13786) recipe. This flag is only "
            "supported when the model has untied word embedding and language modeling head layers i.e. "
            "`tie_word_embeddings` in the model config is False."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    # 自动移除数据集中未被训练流程直接使用的列。如果你的自定义奖励函数需要使用数据集中的其他列（除了 'prompts' 和 'completions'），
    # 你应该将此参数设置为 False，以确保这些列不会被删除，从而避免在计算奖励时出现错误。
    
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )

    # rollout 次数，有效batchsize = num_processes * per_device_train_batch_size * gradient_accumulation_steps 必须被这个数整除。
    num_generations: int | None = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )

    # 一个prompt在评估时采样的回答数。如果设置为 None，则在评估时使用与训练时相同的采样数。
    # 这允许在评估时使用较少的采样数以节省计算资源。
    num_generations_eval: int | None = field(
        default=None,
        metadata={
            "help": "Number of generations to sample during evaluation. This allows using fewer generations during "
            "evaluation to save computation. If `None`, uses the value of `num_generations`."
        },
    )

    # 生成回答时的最大长度。
    max_completion_length: int | None = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )

    # 针对zero3的优化选项
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )

    # 是否在每个训练 epoch 开始时对训练数据集进行洗牌（shuffle）。
    shuffle_dataset: bool | None = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Parameters that control generation
    # 生成回答时的批量大小。如果设置为 None，则默认为有效训练批量大小：per_device_train_batch_size * num_processes * steps_per_generation
    generation_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * steps_per_generation`."
        },
    )

    """
    steps_per_generation = gradient_accumulation_steps（默认值）
    每次参数更新后，立即用新策略生成新数据。
    优点：策略始终基于最新模型采样，数据分布与当前策略一致（on-policy）。
    缺点：生成开销大（频繁调用模型 + 奖励函数）
    steps_per_generation > gradient_accumulation_steps
    表示：用同一批生成数据，更新 2 次参数（8 ÷ 4 = 2）。
    效果：类似 off-policy 训练（重用旧数据）。
    优点：减少生成次数，节省计算（尤其当 RM 很慢时）。
    风险：如果策略变化快，旧数据可能过时 → 训练不稳定。
        适用场景：当奖励函数计算开销大，或生成速度慢时（大模型、多模态模型），可考虑设置更大值。
    steps_per_generation < gradient_accumulation_steps
    通常不合法或无意义，因为一次参数更新还没完成就要新数据。
    大多数实现会禁止此情况（或自动对齐到 ≥ gradient_accumulation_steps）。
    """
    steps_per_generation: int | None = field(
        default=None,
        metadata={"help": "Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`."},
    )

    # 生成回答时的随机性控制参数
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )

    # 控制生成回答时的多样性, 从概率最高的 token 开始，累加概率直到总和 ≥ p，然后只在这个“最小可行集合”中采样。
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )

    # 控制生成回答时的多样性, 只在概率最高的 k 个 token 中采样。
    top_k: int | None = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )

    # 生成回答时的最小概率阈值，防止采样过于集中在少数高概率 token 上。
    min_p: float | None = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )

    # 额外的生成参数
    generation_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or "
            "`SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the "
            "generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that "
            "conflict with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them."
        },
    )

    # 额外的参数，用于在生成回答时应用聊天模板（chat template）
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating "
            "completions."
        },
    )

    """
    repetition_penalty 用于在生成过程中惩罚重复出现的 token。
    模型输出原始 logits（未归一化的分数）。
    对于已经在 prompt 或已生成文本中出现过的 token，其 logit 会被除以 repetition_penalty。
    然后再进行 softmax 或采样。
    """
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )

    """
    所有序列在同一个 batch 中被 padding 到相同长度；
    即使某些序列很短，也会为它们分配与最长序列相同的 KV cache 内存；
    导致 显存浪费严重，尤其在生成长度差异大时（如对话数据）。
    💡 Paged Attention（由 vLLM 首创）的思想是：

    将每个序列的 KV cache 像操作系统分页一样，划分为固定大小的块（blocks）；
    只为实际生成的 token 分配内存；
    显存利用率大幅提升，支持更高吞吐和更长上下文。
    """
    use_transformers_paged: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the `transformers` paged implementation for generation. If set to `True`, the "
            "`transformers` paged implementation will be used for generation instead of the default padded "
            "implementation. This parameter is only effective when `use_vllm` is set to `False`."
        },
    )

    # KV cache 加速生成的缓存方法实现
    cache_implementation: str | None = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )

    # Parameters that control generation acceleration powered by vLLM
    # 是否启用 vLLM 进行加速生成。
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure "
            "a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: "
            "Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for "
            "model implementation."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory "
            "usage low, but waking the engine adds host–device transfer latency."
        },
    )
    vllm_guided_decoding_regex: str | None = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

    # Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)
    vllm_server_base_url: str | None = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` "
            "and `vllm_server_port` are ignored."
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )

    # Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_max_model_length: int | None = field(
        default=None,
        metadata={
            "help": "Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus "
            "`max_completion_length`; if omitted, it is inferred from the model config."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
        },
    )

    # Parameters that control the training
    # GRPO中控制KL散度惩罚项的系数。如果设置为0.0，则不使用参考模型，从而节省内存并加快训练速度， deepseek-r1论文建议使用0.001。
    beta: float = field(
        default=0.001,
        metadata={
            "help": "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and "
            "improving training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement "
            "learning](https://huggingface.co/papers/2501.12948) use a value of `0.001`."
        },
    )

    # 类似于PPO epoches
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )

    # 单边裁剪的  epsilon 值
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )

    # 双边裁剪的上限值 delta
    delta: float | None = field(
        default=None,
        metadata={
            "help": "Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` "
            "(default), standard GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This "
            "method is introduced in the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)."
        },
    )

    # clip的上限值
    epsilon_high: float | None = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`. "
            "When used with `loss_type='cispo'`, this corresponds to the ε_max param specified in the"
            "[ScaleRL paper]https://huggingface.co/papers/2510.13786) and the recommended value is `5.0`."
        },
    )
    sapo_temperature_neg: float = field(
        default=1.05,
        metadata={
            "help": "Temperature for tokens with non-positive advantage scores used in the `sapo` loss function. "
            "This parameter is introduced in the [Soft Adaptive Policy Optimization "
            "paper](https://huggingface.co/papers/2511.20347)."
        },
    )
    sapo_temperature_pos: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for tokens with positive advantage scores used in the `sapo` loss function. "
            "This parameter is introduced in the [Soft Adaptive Policy Optimization "
            "paper](https://huggingface.co/papers/2511.20347)."
        },
    )

    """
    这个参数 importance_sampling_level 控制 在 GRPO（或其变体如 GSPO）训练中，重要性采样比率（importance sampling ratio）
    是在 token 级别还是 sequence 级别计算的。它直接影响策略梯度的归一化方式和训练稳定性。
    """
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. "
            "`'token'` keeps the raw per-token log-probability ratios (one weight per token).  `'sequence'` averages "
            "the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper "
            "shows that sequence-level sampling often yields more stable training and better alignment with "
            "sequence-level rewards."
        },
    )

    # 奖励函数的权重
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )

    """
    在强化学习中：
    奖励的绝对大小会影响策略梯度的幅度；
    如果 reward 方差过大（如某些样本得 100 分，某些得 0.1 分），优化会不稳定；
    标准化（如除以标准差）可使 reward 具有“单位方差”，提升训练鲁棒性。
    """
    scale_rewards: str = field(
        default="group",
        metadata={
            "help": "Specifies the scaling strategy for rewards. Supported values are: "
            "`True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring "
            "unit variance within a group. "
            "`'batch'`: rewards are scaled by the standard deviation across the entire batch, as recommended in the "
            "PPO Lite paper. "
            "`False` or `'none'`: no scaling is applied. The Dr. GRPO paper recommends not scaling rewards, as "
            "scaling by the standard deviation introduces a question-level difficulty bias."
        },
    )

    # 损失函数的类型
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are 'grpo', 'dapo', 'bnpo', and "
            "'dr_grpo'. "
            "'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length "
            "bias—this approach tends to prefer shorter completions with positive advantages and longer ones with "
            "negative advantages. "
            "'dapo' (default): Aggregates token-level losses by normalizing with the number of active token in the "
            "global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias. "
            "'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`. "
            "'bnpo': Aggregates token-level losses by normalizing with the number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss."
            "'cispo': Clips the importance sampling weights instead of the advantage scaled importance weights. "
            "The clipped weights are then multiplied with the advantages and policy model's log probs. "
            "Individual token losses are aggregated by normalizing with the number of active tokens in "
            "the global accumulated batch. This method was introduced in the "
            "[MiniMax-M1 paper](https://huggingface.co/papers/2506.13585)."
            "'sapo': Soft Adaptive Policy Optimization loss, as introduced in the "
            "[Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2506.13585). "
            "Replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates "
            "off-policy updates while preserving useful learning signals."
        },
    )

    """
    在计算 loss 时，自动忽略因达到最大长度（max_completion_length）而被截断（truncated）的 completion 的末尾部分，
    避免对“非模型本意”的截断 token 进行错误惩罚。
    """
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )

    """
    在标准 RLHF/GRPO 中：
    参考模型（reference model） 通常是初始的 SFT 模型，权重固定不变；
    它的作用是提供 KL 散度约束：防止策略模型偏离太远；
    但问题在于：
    随着策略模型不断优化，它与固定参考模型的差距越来越大；
    KL penalty 会越来越强，抑制进一步学习（“KL collapse”）；
    尤其在长训练或高 reward 任务中，模型可能被“锁死”。
    ✅ 解决方案：让参考模型缓慢跟随策略模型更新，保持两者“适度接近”，既保留约束作用，又不阻碍进步。
    """
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )

    # 参考模型与当前策略模型混合的比例 α
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    # 参考模型同步的频率 τ
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    # 只让模型从“高不确定性（高熵）”的 token 位置学习策略梯度，忽略低熵（确定性强）的位置，从而提升训练效率与对齐质量。
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={
            "help": "ρ parameter from Beyond the 80/20 Rule. Keeps in the policy loss term only the top-ρ quantile of "
            "tokens by entropy of the probability distribution at each sequence position, improving results. Range: "
            "[0.0-1.0]. A value of `0.0` masks all but the highest entropy token; `1.0` keeps all tokens. The paper "
            "recommends a value of `0.2`. If used with `mask_truncated_completions=True`, only tokens from "
            "non-truncated completions are considered."
        },
    )

    """
    use_liger_loss 是一个用于启用 Liger Kernel 优化版 GRPO loss 的开关参数。
    它的核心目的是：通过高度优化的 CUDA 内核（Liger Kernel）加速训练、降低显存占用，并提升数值稳定性
    """
    use_liger_loss: bool = field(
        default=None,
        metadata={"help": "Whether to use the Liger GRPO loss."},
    )
    vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply Importance Sampling (IS) to correct for the mismatch between vLLM "
            "completion logprobs and recomputed training logprobs. If set to `False`, no IS is applied "
            "regardless of `vllm_importance_sampling_mode`. When `True`, the selected mode determines how "
            "IS ratios are computed and constrained."
        },
    )

    vllm_importance_sampling_mode: str = field(
        default="sequence_mask",
        metadata={
            "help": "Specifies how Importance Sampling (IS) is performed when "
            "vllm_importance_sampling_correction=True. Modes are defined along two orthogonal "
            "dimensions: (1) constraint, which determines how to handle ratios above "
            "vllm_importance_sampling_cap (C)—either truncation (clip from above, ρ ← min(ρ, C)) or "
            "masking (set ratios above C to zero); and (2) granularity, which determines whether "
            "ratios are computed per token or as a single sequence-level ratio applied to all tokens. "
            "Supported options are: 'token_truncate', 'token_mask', 'sequence_truncate', and "
            "'sequence_mask'."
        },
    )

    vllm_importance_sampling_cap: float = field(
        default=3.0,
        metadata={
            "help": "Importance sampling cap C used by `vllm_importance_sampling_mode`. For '*_truncate' modes, "
            "ratios are clipped from above at C. For '*_mask' modes, ratios larger than C are set to zero."
        },
    )

    # 修正off-policy KL散度估计的偏差
    use_bias_correction_kl: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the unbiased KL divergence estimator with importance sampling correction. This "
            "corrects the KL divergence estimate by multiplying it with the importance sampling ratio. "
            "This is described in the [DeepSeek-V3.2 paper](https://huggingface.co/papers/2512.02556)."
        },
    )

    # Parameters that control the logging
    """
    log_completions 是一个用于调试和监控训练过程的实用功能开关。当启用时，它会在训练过程中定期（每 logging_steps 步）
    记录并展示一批当前策略模型生成的 (prompt, completion) 样本，帮助你直观判断：
    模型是否在“胡说八道”？
    是否学会了遵循指令？
    生成长度、格式、风格是否合理？
    是否出现退化（如重复、截断、乱码）？
    """
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )

    # 每次记录时打印的回答数量
    num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )

    # 决定在记录 (prompt, completion) 样本时，是否对 prompt 进行去重（只保留唯一 prompt）。
    log_unique_prompts: bool = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all "
            "prompts are logged."
        },
    )

    # Deprecated arguments
    # prompt 最大长度，建议在训练前过滤数据集以确保 prompt 不超过所需长度。
    max_prompt_length: int | None = field(
        default=None,
        metadata={
            "help": "Deprecated, filter your dataset before training to ensure that prompts do not exceed your "
            "desired length."
        },
    )
    wandb_log_unique_prompts: bool | None = field(
        default=None,
        metadata={"help": "Deprecated, use `log_unique_prompts` instead."},
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Determine the number of generations to use for evaluation
            num_generations = self.num_generations_eval or self.num_generations

            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        # if self.num_generations < 2:
        #     raise ValueError(
        #         "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
        #         f"{self.num_generations}, which is less than the minimum required."
        #     )

        if self.use_liger_loss is not None:
            warnings.warn(
                "The `use_liger_loss` argument is deprecated and will be removed in version 0.28.0. Please use "
                "`use_liger_kernel` instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.use_liger_kernel = self.use_liger_loss

        if self.delta is not None and self.use_liger_kernel:
            raise ValueError("Liger kernel does not support two-sided GRPO loss yet.")

        if self.max_prompt_length is not None:
            warnings.warn(
                "The `max_prompt_length` argument is deprecated and will be removed in version 0.28.0. You should "
                "instead filter your dataset before training to ensure that prompts do not exceed your desired "
                "length.",
                FutureWarning,
                stacklevel=2,
            )

        if self.wandb_log_unique_prompts is not None:
            warnings.warn(
                "The `wandb_log_unique_prompts` argument is deprecated and will be removed in version 0.27.0. Please "
                "use `log_unique_prompts` instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.log_unique_prompts = self.wandb_log_unique_prompts
