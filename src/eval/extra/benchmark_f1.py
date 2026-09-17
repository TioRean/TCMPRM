
import pandas as pd
import re
def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"reading Excel fail: {e}")
        return None

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


def clean_answer(ans):
    if pd.isna(ans):
        return []
    ans_str = str(ans).strip().upper()
    return sorted(list(set([c for c in ans_str if c in 'ABCDE'])))

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

def calc_benchmark_f1(resp_output_path, f1_output_path):
    df = read_excel(resp_output_path)
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    for i, row in df.iterrows():
        try:
            pred = extract_answer(row['response'])
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
    calc_benchmark_f1(
        resp_output_path="D:\桌面文件\\benchmark结果\\benchmark-case_sft_model-all-qa-CH.xlsx",
        f1_output_path="D:\桌面文件\\benchmark结果\\benchmark-case_sft_model-all-qa-CH.txt",
    )
