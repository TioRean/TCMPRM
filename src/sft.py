
import os

DS_CONFIG = "ds_zero2_no_offload.json"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
os.environ["TOKENIZER_PARALLELISM"] = "false"
# os.environ['VLLM_USE_MODELSCOPE'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from trl import SFTTrainer, SFTConfig
import os
import json
import torch
from datasets import Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from swanlab.integration.transformers import SwanLabCallback


def format_with_positions(conversation, tokenizer):
    formatted_parts = []
    answer_spans = []
    current_total_length = 0

    for msg in conversation:
        role, content = msg["role"], msg["content"]
        if role == "system":
            part = f"<|im_start|>system\n{content}<|im_end|>\n"
            formatted_parts.append(part)
            current_total_length += len(part)
        elif role == "user":
            part = f"<|im_start|>user\n{content}<|im_end|>\n"
            formatted_parts.append(part)
            current_total_length += len(part)
        elif role == "assistant":
            assistant_prefix = "<|im_start|>assistant\n"
            assistant_suffix = "<|im_end|>\n"
            full_assistant_block = assistant_prefix + content + assistant_suffix

            ans_start = current_total_length
            ans_end = current_total_length + len(full_assistant_block)

            formatted_parts.append(full_assistant_block)
            current_total_length += len(full_assistant_block)
            answer_spans.append((ans_start, ans_end))

    return "".join(formatted_parts), answer_spans


def map_char_spans_to_token_spans(text, char_spans, tokenizer, max_len=None):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoding.offset_mapping

    token_spans = []
    for char_s, char_e in char_spans:
        t_s, t_e = None, None
        for idx, (s, e) in enumerate(offsets):
            if t_s is None and s <= char_s < e:
                t_s = idx
            if t_e is None and s < char_e <= e:
                t_e = idx + 1
                break

        if t_s is not None and t_e is not None:
            if max_len is not None:
                t_s = min(t_s, max_len - 1)
                t_e = min(t_e, max_len)
            if t_s < t_e:
                token_spans.append((t_s, t_e))
    return token_spans


def batched_preprocess_function(examples, tokenizer, max_length=5200):
    all_texts = []
    all_spans = []

    for messages in examples["messages"]:
        txt, spans = format_with_positions(messages, tokenizer)
        all_texts.append(txt)
        all_spans.append(spans)

    encodings = tokenizer(
        all_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True
    )

    batch_spans = []
    for t, s in zip(all_texts, all_spans):
        spans = map_char_spans_to_token_spans(t, s, tokenizer, max_len=max_length)
        batch_spans.append(spans)

    labels = torch.full_like(encodings.input_ids, -100)
    for i, spans in enumerate(batch_spans):
        for t_s, t_e in spans:
            if t_s >= max_length or t_e > max_length:
                continue
            labels[i, t_s:t_e] = encodings.input_ids[i, t_s:t_e]

    return {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
        "labels": labels
    }
model_name = 'Qwen/Qwen2.5-7B-Instruct'

data_path_case = os.path.join('/src/data/case-cot.json')
data_path_qa = os.path.join('/src/data/qa-cot.json')
MAX_LENGTH = 4000

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


with open(data_path_case, 'r', encoding='utf-8') as f:
    case_dataset = Dataset.from_list(json.load(f))

case_dataset = case_dataset.map(
    lambda x: batched_preprocess_function(x, tokenizer, max_length=MAX_LENGTH),
    batched=True,
    remove_columns=["messages"]
)

with open(data_path_qa, 'r', encoding='utf-8') as f:
    qa_dataset = Dataset.from_list(json.load(f))

qa_dataset = qa_dataset.map(
    lambda x: batched_preprocess_function(x, tokenizer, max_length=MAX_LENGTH),
    batched=True,
    remove_columns=["messages"]
)

dataset = concatenate_datasets([case_dataset, case_dataset, qa_dataset])

split_dataset = dataset.train_test_split(test_size=0.025, seed=42, shuffle=True)

swanlab_callback = SwanLabCallback(
    project="tcm-model",
    experiment_name="qwen_alldata-cot",
    description="qwen cot LoRA sft with alldata-cot",
    config={
        "model": "qwen",
        "dataset": "alldata-cot",
        "lora_rank": 64,
        "lora_alpha": 128,
        "streaming_load": True,
    },
)


sft_config = TrainingArguments(
    output_dir="/mnt/data3/",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=10,
    warmup_steps=600,
    num_train_epochs=1,
    logging_steps=20,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    weight_decay=0.01,
    seed=3407,
    report_to="none",
    bf16=True,
    max_grad_norm=0.5,
    deepspeed=DS_CONFIG,
    logging_first_step=True,
    per_device_eval_batch_size=10,
    eval_strategy='steps',
    eval_steps=2000,
    save_strategy='steps',
    save_steps=1950,
    save_total_limit=30,
    remove_unused_columns=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=False,
)


trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    callbacks=[swanlab_callback],
    args=sft_config,
)

trainer.train()

save_dir = "/src/model/"
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)
