
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re
from typing import List, Dict
import torch
import json
import os
class TCMPrescriptionRater:
    TCM_UNITS = ['g', '克', '两', '钱', '斤', 'ml', '毫升', '片', '枚', '根', '条', '粒', '合', '升', '寸']

    TCM_METHODS = [
        "煎汤代水", "沸水浸", "去上沫",
        "先煎", "后下", "包煎", "布包", "冲服", "烊化", "另煎", "兑服",
        "焗服", "泡服", "炖服", "布包", "捣碎", "研末", "磨粉",
        "打碎", "掰开", "去核", "去壳", "渍服", "冷浸", "温浸", "煮散",
        "蜜炙", "酒炙", "醋炙", "姜炙", "盐炙",
        "醋炒", "酒炒", "盐炒", "土炒", "麸炒", "清炒",
        "酒制", "醋制", "姜制", "盐制", "蜜制",
        "煅制", "蒸制", "煮制", "煨制", "发酵", "水飞",
        "炒", "炙", "煅", "煨", "蒸", "煮", "炮", "制",
        "碎", "掰", "冲", "包", "浸", "渍", "泡", "焗",
    ]

    MIN_PRESCRIPTION_UNITS = 1
    SENTENCE_SPLITTERS = r'。|！|？|；|\n\n'

    def __init__(self, med_std_tab:dict):
        self.TCM_METHODS = sorted(self.TCM_METHODS, key=len, reverse=True)
        self.method_list = '|'.join(re.escape(m) for m in self.TCM_METHODS)
        self.unit_list = '|'.join(re.escape(u) for u in self.TCM_UNITS)

        self.PATTERN = re.compile(
            r'([\u4e00-\u9fa5]{2,6})'
            r'[：:;\s、,\-~—　]*'
            r'(\d+\.?\d*)'
            r'[：:;\s、,\-~—　]*'
            fr'({self.unit_list})'
            r'[：:;\s、,\-~—　]*'
            r'(?:\(|（|【|〔)?'
            fr'({self.method_list})?'
            r'(?:\)|）|】|〕)?'
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.std_med_tab = dict(sorted(med_std_tab.items(), key=lambda x: len(x[0]), reverse=True))

    def standardize_med(self, x):
        return self.std_med_tab[x] if x in self.std_med_tab else x

    def extract_tcm_prescription_sentences(self, text: str, output_longest: bool = True, standardize_med = True) -> List[Dict]:
        if not isinstance(text, str):
            return []
        if 'Final Response' in text:
            text = text.split('Final Response')[-1]
        text = text.replace('**', '')
        sentences = re.split(self.SENTENCE_SPLITTERS, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        result = []
        for sent in sentences:
            matches = self.PATTERN.findall(sent)
            valid = [m for m in matches if m[0] and m[1] and m[2]]
            if len(valid) >= self.MIN_PRESCRIPTION_UNITS:
                item = {
                    "sentence": sent,
                    "ingredient_count": len(valid),
                    "med": [],
                    "med+dose": ''
                }
                for dname, dose, unit, method in valid:
                    if standardize_med:
                        dname = self.standardize_med(dname)
                    item["med"].append(dname)
                    item['med+dose'] += dname + dose + unit + ' '
                result.append(item)

        if result and output_longest:
            result.sort(key=lambda x: x["ingredient_count"], reverse=True)
            result = [result[0]]

        return result

    def extract_psc(self, text: str, output_longest: bool = True, standardize_med:bool=True) -> Dict:
        res = self.extract_tcm_prescription_sentences(text, output_longest, standardize_med)
        try:
            return res[0]
        except:
            return {
                    "sentence": '',
                    "ingredient_count": 0,
                    "med": [],
                    "med+dose": ''
                }

std_med_tab_path = '/qwen/std_med_tab.json'
with open(std_med_tab_path, 'r', encoding='utf-8') as f:
    std_med_tab = json.load(f)

tcm_extractor = TCMPrescriptionRater(std_med_tab)

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

def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"reading Excel failed: {e}")
        return None

def create_prompt(question):
    template = '根据以下医案信息，给出一个包含药物组成、剂量的中医处方。{}。'
    content = template.format(question)
    return [{"role": "user", "content": content}]

def process_questions_batch(case_eval_path, model, tokenizer, resp_output_path, batch_size=8):
    df = read_excel(case_eval_path)
    if df is None:
        return
    df["response"] = ""
    df["extracted_prescription"] = ""
    all_messages = []
    total = len(df)

    for i in range(total):
        q = df.loc[i, "Clinical case (CH)"]
        msg = create_prompt(q)
        all_messages.append(msg)

    print(f"Total：{total}，batch_size：{batch_size}")

    for i in tqdm(range(0, total, batch_size), desc="推理中"):
        end = min(i + batch_size, total)
        batch_msg = all_messages[i:end]
        indices = list(range(i, end))
        try:
            prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msg]
            with torch.no_grad():
                outputs = model.generate(prompts, sampling_params)

            for idx, df_idx in enumerate(indices):
                resp = outputs[idx].outputs[0].text
                df.loc[df_idx, "response"] = resp

                pred = tcm_extractor.extract_psc(resp)
                df.loc[df_idx, "extracted_prescription"] = pred['med+dose']
        except Exception as e:
            print(f"batch {i}-{end} err: {e}")

    df.to_excel(resp_output_path, index=False)
    print(f"The responses were saved to: {resp_output_path}")


def calc_case_f1(resp_output_path, f1_output_path):
    df = read_excel(resp_output_path)
    if df is None:
        return
    g_total_tp = 0.0
    g_total_fp = 0.0
    g_total_fn = 0.0
    t_total_tp = 0.0
    t_total_fp = 0.0
    t_total_fn = 0.0

    for i, row in df.iterrows():
        try:
            pred = tcm_extractor.extract_psc(row["extracted_prescription"])
            org = tcm_extractor.extract_psc(row["Original Medicine ingredients (CH)"])
            g_tp, g_fp, g_fn = calc_metrics(pred['med'], org['med'])
            g_total_tp += g_tp
            g_total_fp += g_fp
            g_total_fn += g_fn

            prefer = tcm_extractor.extract_psc(row["Preferred Medicine ingredients (CH)"])
            t_tp, t_fp, t_fn = calc_metrics(pred['med'], prefer['med'])
            t_total_tp += t_tp
            t_total_fp += t_fp
            t_total_fn += t_fn
        except Exception as e:
            pass
    gold_micro_total_f1 = calc_f1_score(g_total_tp, g_total_fp, g_total_fn)
    target_micro_total_f1 = calc_f1_score(t_total_tp, t_total_fp, t_total_fn)

    result = f'''
    original_micro_total_f1: {gold_micro_total_f1}
    preferred_micro_total_f1: {target_micro_total_f1}
    '''
    print(result)
    with open(f1_output_path, 'w', encoding='utf-8') as result_file:
        result_file.write(result)
    print(f"The f1 result was saved to: {f1_output_path}")

if __name__ == "__main__":
    model_path = "/qwen/model/"
    model_dir_name = os.path.basename(os.path.normpath(model_path))
    case_eval_path = "./Gen/case_eval.xlsx"
    resp_output_path = f"./Gen/case_eval_{model_dir_name}.xlsx"
    f1_output_path = f"./Gen/case_eval_{model_dir_name}.txt"

    BATCH_SIZE = 100
    llm_config = {
        "max_num_seqs": BATCH_SIZE,
        "max_model_len": 4096,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.95,
    }

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.9,
        max_tokens=4096,
        repetition_penalty=1.05
    )

    llm = LLM(model=model_path, **llm_config)
    tokenizer = llm.get_tokenizer()

    process_questions_batch(
        case_eval_path=case_eval_path,
        model=llm,
        tokenizer=tokenizer,
        resp_output_path=resp_output_path,
        batch_size=BATCH_SIZE
    )

    calc_case_f1(resp_output_path, f1_output_path)