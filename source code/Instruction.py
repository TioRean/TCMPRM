
import os
DS_CONFIG = "ds_zero2_no_offload.json"
os.environ['OMP_NUM_THREADS'] = '1'
# A100 是8.0
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['VLLM_USE_MODELSCOPE'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# import deepspeed

def format_with_positions(conversation):
    """
    格式化对话，并记录assistant回答的字符位置范围。

    Args:
        conversation: List[dict], 多轮对话，例如:
            [
                {"role": "user", "content": "解释机器学习"},
                {"role": "assistant", "content": "机器学习是..."}
            ]

    Returns:
        formatted_str: str, 格式化后的字符串
        answer_start: int, assistant回答的起始字符位置
        answer_end: int, assistant回答的结束字符位置（不包括）
    """
    formatted_parts = []
    answer_start, answer_end = None, None

    for msg in conversation:
        role, content = msg["role"], msg["content"]
        if role == "system":
            formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            answer_start = len("".join(formatted_parts)) + len("<|im_start|>assistant\n")
            formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
            answer_end = answer_start + len(content)

    formatted_str = "".join(formatted_parts)
    return formatted_str, answer_start, answer_end

def map_char_positions_to_token_positions(
        text, char_start, char_end, tokenizer
):

    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]  # List of (char_start, char_end) for each token

    token_start, token_end = None, None

    for i, (start, end) in enumerate(offsets):
        if start <= char_start < end:
            token_start = i
        if start <= char_end <= end:
            token_end = i + 1
            break

    return token_start, token_end

def build_labels(input_ids, token_start, token_end):

    labels = input_ids.clone()
    labels[:] = -100  # 默认全部忽略
    labels[0, token_start:token_end] = input_ids[0, token_start:token_end]  # 只保留回答部分
    return labels


class PreTokenizedDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        }

        return batch
