import os

DS_CONFIG = "ds_zero2_no_offload.json"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['VLLM_USE_MODELSCOPE'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset, Dataset
from swanlab.integration.transformers import SwanLabCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, get_peft_model
import json
import os
# import deepspeed
from Instruction import PreTokenizedDataCollator
import Instruction

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)

model.enable_input_require_grads()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=128,
    lora_alpha=256,
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

def batched_preprocess_function(examples):
    formatted_strs, answer_starts, answer_ends = zip(*[
        Instruction.format_with_positions(messages) for messages in examples["messages"]
    ])

    encodings = tokenizer(
        formatted_strs,
        return_tensors="pt",
        padding="max_length",
        max_length=2800,
        truncation=True,
        return_offsets_mapping=True,
    )

    token_ranges = [
        Instruction.map_char_positions_to_token_positions(text, start, end, tokenizer)
        for text, start, end in zip(formatted_strs, answer_starts, answer_ends)
    ]

    labels = torch.full_like(encodings["input_ids"], -100)
    for i, (token_start, token_end) in enumerate(token_ranges):
        if token_start is not None and token_end is not None:
            labels[i, token_start:token_end] = encodings["input_ids"][i, token_start:token_end]

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }

train_data_path = os.path.join('Gen', 'train.jsonl')
val_data_path = os.path.join('Gen', 'val.jsonl')

with open(train_data_path, 'r', encoding='utf-8') as train_file:
    train_ds = Dataset.from_list([json.loads(line.strip()) for line in train_file])
train_ds = train_ds.map(
    batched_preprocess_function,
    batched=True,
    batch_size=50,
    remove_columns=["messages"]
)

with open(val_data_path, 'r', encoding='utf-8') as val_file:
    val_ds = Dataset.from_list([json.loads(line.strip()) for line in val_file])
val_ds = val_ds.map(
    batched_preprocess_function,
    batched=True,
    batch_size=50,
    remove_columns=["messages"]
)

data_collator = PreTokenizedDataCollator(
    tokenizer=tokenizer,
    mlm=False,
)

swanlab_callback = SwanLabCallback(
    project="TCM-model-simulated-only",
    experiment_name="Qwen2.5-7B_finetune",
    description="使用通义千问Qwen2.5-7B在模拟医案数据上微调。",
    config={
        "model": "Qwen2.5-7B",
        "dataset": "模拟医案",
        "train_data_number": len(train_ds),
        "lora_rank": 128,
        "lora_alpha": 256,
        "lora_dropout": 0.1,
    }
)

sft_config = SFTConfig(
        output_dir="./lora_model",
        per_device_train_batch_size = 5,
        gradient_accumulation_steps = 6,
        warmup_steps = 500,
        num_train_epochs = 5.0,
        max_steps= 24200,
        logging_steps = 20,
        optim = "adamw_torch",
        learning_rate = 4e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        report_to = "none",
        bf16=True,
        max_grad_norm=1.0,
        deepspeed=DS_CONFIG,
        logging_first_step=True,
        per_device_eval_batch_size=16,
        eval_accumulation_steps=1,
        eval_strategy = 'steps',
        eval_steps = 2000,
        save_strategy = 'steps',
        save_steps = 2000,
        save_total_limit = 10,
        label_names=["labels"],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory = False,
        overwrite_output_dir= False,
    )

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

trainer = SFTTrainer(
    model = model,
    processing_class=tokenizer,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    callbacks=[swanlab_callback],
    args = sft_config,
    data_collator=data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)


trainer_stats = trainer.train()

