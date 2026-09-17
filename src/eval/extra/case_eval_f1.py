import pandas as pd
import re
from typing import List, Dict
import torch
import json

import os
import glob
import pandas as pd
from openpyxl import load_workbook


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

    def __init__(self, med_std_tab: dict):
        # 按长度倒序排序（避免短匹配优先覆盖长匹配）
        self.TCM_METHODS = sorted(self.TCM_METHODS, key=len, reverse=True)

        # 生成正则匹配列表
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

    def extract_tcm_prescription_sentences(self, text: str, output_longest: bool = True, standardize_med=True) -> List[
        Dict]:
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

    def extract_psc(self, text: str, output_longest: bool = True, standardize_med: bool = True) -> Dict:
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

std_med_tab_path = 'std_med_tab.json'
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


def calc_case_f1(resp_output_path, f1_output_path=None):
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

            gold = tcm_extractor.extract_psc(row["Gold Medicine ingredients (CH)"])
            g_tp, g_fp, g_fn = calc_metrics(pred['med'], gold['med'])
            g_total_tp += g_tp
            g_total_fp += g_fp
            g_total_fn += g_fn

            target = tcm_extractor.extract_psc(row["Target Medicine ingredients (CH)"])
            t_tp, t_fp, t_fn = calc_metrics(pred['med'], target['med'])
            t_total_tp += t_tp
            t_total_fp += t_fp
            t_total_fn += t_fn
        except Exception as e:
            pass
    gold_micro_total_f1 = calc_f1_score(g_total_tp, g_total_fp, g_total_fn)
    target_micro_total_f1 = calc_f1_score(t_total_tp, t_total_fp, t_total_fn)

    result = f'''
    gold_micro_total_f1: {gold_micro_total_f1}
    target_micro_total_f1: {target_micro_total_f1}
    '''
    print(result)
    if f1_output_path:
        with open(f1_output_path, 'w', encoding='utf-8') as result_file:
            result_file.write(result)
        print(f"The f1 result was saved to: {f1_output_path}")


def process_all_excel_files(folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ 错误：文件夹「{folder_path}」不存在，请检查路径！")
        return

    excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    if not excel_files:
        print(f"ℹ️ 提示：文件夹「{folder_path}」中没有找到xlsx文件！")
        return

    print(f"✅ 找到 {len(excel_files)} 个xlsx文件，开始处理...\n")

    for idx, file_path in enumerate(excel_files, 1):
        file_name = os.path.basename(file_path)
        print(f"===== 第 {idx}/{len(excel_files)} 个文件：{file_name} =====")

        result = calc_case_f1(file_path)
        print(result)
        print("\n" + "=" * 80 + "\n")





if __name__ == "__main__":

    folder_path = r""

    process_all_excel_files(folder_path)




