"""统一的多模态大模型推理封装（测试端）。

一个 ``LVLM`` 类通过 ``model_cfg.eval_family`` 适配 LLaVA / Qwen2.5-VL / LLaVA-OneVision，
并按数据集任务（caption / vqa）调用相应的 LASER 打分逻辑。
"""
import warnings

import torch
from PIL import Image

from common.chair_loader import load_chair
from common.registry import eval_ref_logp
from common.vqa_utils import CLEVER_DICT
from evaluate.scorers import caption_scorer, vqa_scorer

warnings.filterwarnings("ignore")

# 训练时统计 logp 所用的特殊 token，测试端保持一致
SPECIAL_TOKEN = "<|image_pad|>"


class LVLM:
    def __init__(self, model_cfg, dataset_cfg, model_path, chair_pkl=None, device="cuda:0"):
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg
        self.family = model_cfg.eval_family
        self.device = device
        self.task_type = dataset_cfg.task_type

        self.ref_logp = eval_ref_logp(model_cfg, dataset_cfg)
        self.beta_sr = dataset_cfg.beta_sr_eval
        self.split_marker = "ASSISTANT: " if self.family == "llava" else "assistant"
        self.caption_mode = "offset" if self.family == "llava" else "find_word"

        self.evaluator = None
        if self.task_type == "caption" and chair_pkl:
            self.evaluator = load_chair(chair_pkl)

        self._build_model(model_path)

    # ------------------------------------------------------------------ build
    def _build_model(self, model_path):
        from transformers import AutoProcessor

        if self.family == "llava":
            from transformers import LlavaForConditionalGeneration

            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                output_attentions=True, output_hidden_states=True, attn_implementation="eager",
            ).to(0)
        elif self.family == "qwen":
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            ).eval().to(self.device)
        elif self.family == "onevision":
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            ).to(0)
        else:
            raise ValueError(f"Unknown eval family: {self.family}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # --------------------------------------------------------------- prepare
    def _prepare_inputs(self, image, question):
        if self.family == "llava":
            conversation = [{
                "role": "user",
                "content": [{"type": "text", "text": question}, {"type": "image"}],
            }]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            return self.processor(images=image, text=prompt, return_tensors="pt").to(0, torch.float16)
        # qwen / onevision
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(0)

    def _special_token_logps(self, inputs, output_ids, prompt_len):
        from trl.trainer.utils import selective_log_softmax

        fw_kwargs = {
            "input_ids": output_ids.unsqueeze(0),
            "attention_mask": torch.ones(1, output_ids.shape[0], device=output_ids.device, dtype=torch.long),
            "return_dict": True,
        }
        if self.family == "llava":
            fw_kwargs["pixel_values"] = inputs.get("pixel_values", None)
        else:
            fw_kwargs["pixel_values"] = inputs["pixel_values"]
            fw_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

        with torch.no_grad():
            fw = self.model(**fw_kwargs)

        logits = fw.logits
        gen_token_ids = output_ids[prompt_len:]
        gen_logits = logits[:, prompt_len - 1:-1, :]
        special_token_id = self.processor.tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN)
        special_token_ids = torch.full_like(gen_token_ids, special_token_id).to(gen_token_ids.device)
        logps = selective_log_softmax(gen_logits, special_token_ids.unsqueeze(0))
        return logps, gen_token_ids

    # ---------------------------------------------------------------- generate
    def generate(self, image, question, args, img_id=None, answer=None, index=None):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self._prepare_inputs(image, question)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=args.max_tokens, do_sample=True,
                temperature=args.inference_temp, return_dict_in_generate=True, output_scores=True,
            )
        output_ids = output.sequences[0]
        logps, gen_token_ids = self._special_token_logps(inputs, output_ids, prompt_len)

        final_ans = self.processor.decode(output_ids, skip_special_tokens=True).split(self.split_marker)[-1].strip()

        if self.task_type == "caption":
            true_scores, false_scores = caption_scorer(
                self.evaluator, img_id, final_ans, gen_token_ids, logps,
                self.processor.tokenizer, self.beta_sr, self.ref_logp, mode=self.caption_mode,
            )
        else:
            if self.dataset_cfg.vqa_candidate_mode == "per_index":
                candidate = CLEVER_DICT[index]
            else:
                candidate = CLEVER_DICT["option"]
            true_scores, false_scores = vqa_scorer(
                final_ans, answer, candidate, gen_token_ids, logps,
                self.processor.tokenizer, self.beta_sr, self.ref_logp,
            )
        return final_ans, true_scores, false_scores
