
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re
import os
def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Reading Excel failed: {e}")
        return None

def clean_answer(ans):
    if pd.isna(ans):
        return []
    ans_str = str(ans).strip().upper()
    return sorted(list(set([c for c in ans_str if c in 'ABCDE'])))

def create_prompt(question, options, q_type):
    if pd.isna(question):
        question = ""

    valid_options = {}
    for k, v in options.items():
        if pd.notna(v) and str(v).strip():
            valid_options[k] = str(v).strip()

    options_text = ""
    for k, v in sorted(valid_options.items()):
        options_text += f"{k}. {v}\n"

    if q_type == 'Single Choice':
        content = f"""分析以下单选题，正确选项字母单独写在最后一行，如A。
        问题：{question}
        选项：
        {options_text}
        """
    elif q_type == 'Multiple Choice':
        content = f"""分析以下多选题，正确选项字母单独写在最后一行，如ABCD。
        问题：{question}
        选项：
        {options_text}
        """
    else:
        raise ValueError(f"不支持的题目类型: {q_type}")

    return [{"role": "user", "content": content}]

import re

def extract_answer(response):
    if not response or not str(response).strip():
        return []

    res = str(response).strip().upper()

    final_key = "## FINAL RESPONSE"
    if final_key in res:
        res = res.split(final_key)[-1].strip()

    pattern_prefix = re.compile(
        r'(?:答案|正确答案|CORRECT ANSWER|ANSWER|选|CHOOSE|THE ANSWER IS)[:：是\s]*([ABCDE,，、\s]+)',
        re.IGNORECASE
    )
    match = pattern_prefix.search(res)
    if match:
        ans_clean = [c for c in match.group(1) if c in 'ABCDE']
        if ans_clean:
            return sorted(list(set(ans_clean)))

    pattern_dot = re.compile(r'([A-E])\.', re.IGNORECASE)
    dot_matches = pattern_dot.findall(res)
    if dot_matches:
        return sorted(list(set(dot_matches)))

    lines = res.split('\n')
    for line in lines:
        cline = line.strip()
        if cline and all(c in 'ABCDE' for c in cline):
            return sorted(list(set(cline)))

    all_letters = re.findall(r'[A-E]', res)
    if all_letters:
        return sorted(list(set(all_letters)))

    return []

def process_questions_batch(benchmark_path, model, tokenizer, resp_output_path, batch_size=10):
    df = read_excel(benchmark_path)
    if df is None:
        return

    required = ["question", "A", "B", "C", "D", "answer"]
    for col in required:
        if col not in df.columns:
            print(f"缺少列: {col}")
            return

    # 新增结果列
    df["response"] = ""

    if "type" not in df.columns:
        df["type"] = "Single Choice"

    all_messages = []
    total = len(df)

    for i in range(total):
        q = df.loc[i, "question"]
        t = df.loc[i, "type"]
        options = {
            "A": df.loc[i, "A"],
            "B": df.loc[i, "B"],
            "C": df.loc[i, "C"],
            "D": df.loc[i, "D"],
        }
        if "E" in df.columns:
            options["E"] = df.loc[i, "E"]

        msg = create_prompt(q, options, t)
        all_messages.append(msg)

    print(f"total：{total}，batch size：{batch_size}")

    for i in tqdm(range(0, total, batch_size), desc="----Reasoning-----"):
        end = min(i + batch_size, total)
        batch_msg = all_messages[i:end]
        indices = list(range(i, end))

        try:
            prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msg]
            outputs = model.generate(prompts, sampling_params)

            for idx, df_idx in enumerate(indices):
                resp = outputs[idx].outputs[0].text
                df.loc[df_idx, "response"] = resp

        except Exception as e:
            print(f"batch {i}-{end} err: {e}")
            for df_idx in indices:
                df.loc[df_idx, "response"] = f"ERROR: {e}"

    df.to_excel(resp_output_path, index=False)
    print(f"The responses were saved to: {resp_output_path}")


def calc_f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    return f1

def calc_metrics(
        pred: list,
        gold: list,
):
    pred_keys = set(pred)
    gold_keys = set(gold)
    tp = len(pred_keys & gold_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    return tp, fp, fn

def calc_benchmark_f1(resp_output_path, f1_output_path):
    df = read_excel(resp_output_path)
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for i, row in df.iterrows():
        try:
            resp = row['response']
            pred = extract_answer(resp)
            gold = clean_answer(row["answer"])
            tp, fp, fn = calc_metrics(pred, gold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        except Exception as e:
            pass

    micro_total_f1 = calc_f1_score(total_tp, total_fp, total_fn)
    result = f'''
        micro_total_f1: {micro_total_f1}
        '''
    print(result)
    with open(f1_output_path, 'w', encoding='utf-8') as result_file:
        result_file.write(result)
    print(f"The f1 result was saved to: {f1_output_path}")



# ====================== 主程序 ======================
if __name__ == "__main__":
    model_path = "/qwen/model/"

    model_dir_name = os.path.basename(os.path.normpath(model_path))

    benchmark_path = "./Gen/benchmark.xlsx"
    resp_output_path = f"./Gen/benchmark-{model_dir_name}.xlsx"
    f1_output_path = f'./Gen/benchmark-{model_dir_name}.txt'

    BATCH_SIZE = 100
    llm_config = {
        "max_num_seqs": BATCH_SIZE,
        "max_model_len": 4096,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.95,
    }

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.90,
        max_tokens=4096,
        repetition_penalty=1.05
    )

    llm = LLM(model=model_path, **llm_config)
    tokenizer = llm.get_tokenizer()
    process_questions_batch(
        benchmark_path=benchmark_path,
        model=llm,
        tokenizer=tokenizer,
        resp_output_path=resp_output_path,
        batch_size=BATCH_SIZE
    )
    calc_benchmark_f1(resp_output_path=resp_output_path, f1_output_path=f1_output_path)
