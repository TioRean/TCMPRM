import copy

import json
import pandas as pd
from itertools import chain
import os
import re
from typing import Union

def save_root_data(TCMRule_path:str, contradictory_path:str):
    _get_root_elems_list(TCMRule_path)
    _get_contradictory_vocab(contradictory_path)

def _get_root_elems_list(TCMRule_path:str):
    def get_elements(dataframe_loc)->list[Union[str, list, set, dict]]:
        if isinstance(dataframe_loc, str):
            return process_elems(dataframe_loc)
        else:
            return [process_elems(x) for x in filter(lambda x: x != 'None', dataframe_loc)]
    def process_elems(x):
        if x == 'None':
            return []
        else:
            content = x.replace('； ', '；').replace('; ', '；').replace(';', '；')
            if has_matching_string(r'<.+>[\(（].+[\)）]', content):
                units_with_parentheses = re.findall(r'<.+>[\(（].+[\)）]', content)
                for unit in units_with_parentheses:
                    match = re.match(r'^(.*?)[\(（](.*)[\)）]', unit)
                    denote = match.group(1)
                    elements = match.group(2).split('；')
                    new_unit = '；'.join([denote + element for element in elements])
                    content = content.replace(unit, new_unit)
            content = content.split('；')
            return content
    def get_elements_list_for_tongue_pulse(dataframe, i):
        trait_groups = (
            ('舌淡白', '舌青'), ('舌老', 'null'), ('白苔', '黑苔'), ('苔少', '苔厚'), ('苔水滑', 'null.1'),
            ('苔腻', 'null.2'), ('脉数', 'null.3'), ('脉促', 'null.4'), ('脉浮', 'null.5'), ('脉大', 'null.6'),
            ('脉虚', 'null.7'), ('脉滑', 'null.8'), ('脉弦', 'null.9'), ('革脉', 'null.10'))
        content = []
        for trait_group in trait_groups:
            traits = get_elements(dataframe.loc[i, trait_group[0]:trait_group[1]])
            if traits == []:
                content.append(['null；；'])
            elif 'F' in traits:
                content.append(['F；；'])
            else:
                for k, trait in enumerate(traits):
                    while len(trait) < 3:
                        trait.append('')
                    traits[k] = '；'.join(trait)
                content.append(traits)
        return content
    def process_elems_with_special_tokens(elem_li):
        elem_dict = dict()
        for elem in elem_li:
            elem = elem.replace(' ', '').replace('《', '<').replace('》', '>').strip(',，;；')
            if re.search(r'<[cC]>', elem):
                denote = re.search(r'<[cC]>', elem).group()
                if denote not in elem_dict.keys():
                    elem_dict[denote] = set()
                elem_dict[denote].add(elem.replace(denote, ''))
            elif re.search(r'<[oO][a-zA-Z]*>', elem):
                denote = re.search(r'<[oO][a-zA-Z]*>', elem).group()
                if denote not in elem_dict.keys():
                    elem_dict[denote] = set()
                elem_dict[denote].add(elem.replace(denote, ''))
            else:
                if '<own>' not in elem_dict.keys():
                    elem_dict['<own>'] = set()
                elem_dict['<own>'].add(elem)
        elem_dict = {k: list(filter(lambda x: x != '' and x != 'null', v)) for k,v in elem_dict.items()}
        elem_dict = {k: v for k, v in elem_dict.items() if len(v) > 0 }
        for k, vs in elem_dict.items():
            maj_elms = [v.replace('<maj>', '') for v in vs if '<maj>' in v]
            vs = [v.replace('<maj>', '') for v in vs]
            if len(maj_elms) > 0 and len(maj_elms) < len(vs):
                elem_dict[k] = {'maj':maj_elms, 'other': list(set(vs) - set(maj_elms))}
            else:
                elem_dict[k] = {'other': vs}
        return elem_dict
    def has_matching_string(pattern, x):
        if isinstance(x, list):
            for s in x:
                if re.search(pattern, s):
                    return True
        if isinstance(x, dict):
            for s in x.keys():
                if re.search(pattern, s):
                    return True
        elif isinstance(x, str):
            if re.search(pattern, x):
                return True
        else:
            raise Exception
        return False
    def _align_time_unit(x):
        tab = {
            '年': {'年': 1, '月': 12, '天': 365, '小时': 8760},
            '岁': {'岁': 1, '月': 12, '天': 365, '小时': 8760},
            '月': {'月': 1, '天': 30, '小时': 720},
            '天': {'天': 1, '小时': 24},
            '小时': {'小时': 1}
        }
        content = []
        for sub_x in x.split('，'):
            elem, range = sub_x.split('：')
            mini, maxi = [pair.split('-') for pair in range.split('->')]
            times = tab[maxi[1]][mini[1]]
            content.append([elem, int(mini[0]), int(maxi[0]) * times, mini[1], maxi[1]]) # 元素 最小值 最大值 最小单位 最大单位
        return content

    df = pd.read_excel(TCMRule_path)
    df = df.fillna('None')
    df_1 = df.loc[:, '方剂编号':'参考资料'].astype(str)
    df_2 = df.loc[:, '舌淡白':'null.10'].astype(str)
    df_3 = df.loc[:, '方名':'加减法查找表L'].astype(str)
    for column in list(df_2.columns):
        if re.match(r'null\.\d+', column):
            temp = 'null'
            df_2.loc[:, column] = df_2[column].str.replace('T', temp, regex=False)
        else:
            df_2.loc[:, column] = df_2[column].str.replace('T', column, regex=False)
    root_elements_list = []
    for i in range(0, len(df)):
        prescription_num = get_elements(df_1.loc[i, '方剂编号'])
        prescription_num = str(prescription_num).replace('[', '').replace(']', '').replace("'", '')

        rule_num = get_elements(df_1.loc[i, '规则编号'])
        rule_num = str(rule_num).replace('[', '').replace(']', '').replace("'", '')

        classification = get_elements(df_1.loc[i, '类别'])
        compulsories = get_elements(df_1.loc[i, '必要元素库A':'必要元素库J'])
        while len(compulsories) < 10:
            compulsories.append([])

        optionals:list[Union[str, list, dict]] = get_elements(df_1.loc[i, '备选元素库A':'备选元素库AD'])
        while len(optionals) < 30:
            optionals.append([])

        sex = process_elems_with_special_tokens(get_elements(df_1.loc[i, '性别']))
        if not has_matching_string(r'<[cC]>', sex) and not has_matching_string(r'<[oO][a-zA-Z]*>', sex):
            if '<own>' not in sex:
                sex['<own>'] = {'other':list()}
            if '男' not in sex['<own>']['other']:
                sex['<own>']['other'].append('男')
            if '女' not in sex['<own>']['other']:
                sex['<own>']['other'].append('女')
        assert len(sex.keys()) == 1
        optionals.append(sex)

        age = process_elems_with_special_tokens(get_elements(df_1.loc[i, '年龄']))
        if not has_matching_string(r'<[cC]>', age) and not has_matching_string(r'<[oO][a-zA-Z]*>', age):
            if '<own>' not in age:
                age['<own>'] = {'other':list()}
            if '成年人：18-岁->65-岁' not in age['<own>']:
                age['<own>']['other'].append('成年人：18-岁->65-岁')

        age = {k: {v_k: [_align_time_unit(v_v) for v_v in v_vs] for v_k, v_vs in v_dict.items()}
               for k, v_dict in age.items()}
        assert len(age.keys()) == 1
        optionals.append(age)

        time_of_onset = df_1.loc[i, '发病时间元素库']
        if not has_matching_string(r'<.+>[\(（].+[\)）]', time_of_onset):
            time_of_onset = f"<OA>({time_of_onset})"
        time_of_onset = process_elems_with_special_tokens(get_elements(time_of_onset))
        time_of_onset = {k: {v_k: [_align_time_unit(v_v) for v_v in v_vs] for v_k, v_vs in v_dict.items()}
               for k, v_dict in time_of_onset.items()}
        assert len(time_of_onset.keys()) == 1
        optionals.append(time_of_onset)

        aggravating = process_elems_with_special_tokens(get_elements(df_1.loc[i, '病情加重元素库']))
        optionals.append(aggravating)
        inducement = process_elems_with_special_tokens(get_elements(df_1.loc[i, '诱因元素库']))
        optionals.append(inducement)
        previous_treatment = process_elems_with_special_tokens(get_elements(df_1.loc[i, '治疗元素库']))
        optionals.append(previous_treatment)
        auxiliary_examination = process_elems_with_special_tokens(get_elements(df_1.loc[i, '辅助检查元素库']))
        optionals.append(auxiliary_examination)
        diagnosis = process_elems_with_special_tokens(get_elements(df_1.loc[i, '疾病元素库']))
        optionals.append(diagnosis)
        past_med_history = process_elems_with_special_tokens(get_elements(df_1.loc[i, '既往史元素库']))
        optionals.append(past_med_history)
        shared = get_elements(df_1.loc[i, '共享元素库'])
        optionals.append(shared)
        negative = process_elems_with_special_tokens(get_elements(df_1.loc[i, '阴性元素库']))
        for K, V in negative.items():
            for k, v in V.items():
                negative[K][k] = ['无' + x for x in v]
        note = get_elements(df_1.loc[i, '注意事项'])
        conference = get_elements(df_1.loc[i, '参考资料'])
        tongue_pulse = get_elements_list_for_tongue_pulse(dataframe=df_2, i=i)
        prescription = get_elements(df_3.loc[i, '方名'])
        medicines = get_elements(df_3.loc[i, '药物1':'药物50'])
        mdf_lookup_table = df_3.loc[i, '加减法查找表A':'加减法查找表L'].to_dict()
        mdf_lookup_table = [json.loads(v) for v in mdf_lookup_table.values() if v != 'None']

        root_elements_list.append(
            [prescription_num, rule_num, classification, compulsories, optionals, negative, tongue_pulse,
             prescription, medicines, note, conference, mdf_lookup_table])
    json_path = os.path.splitext(TCMRule_path)[0] + '.json'
    with open(json_path, 'w') as f:
        content = json.dumps(root_elements_list, ensure_ascii=False)
        f.write(content)
    return root_elements_list


def _get_contradictory_vocab(contradictory_path:str):
    df = pd.read_excel(contradictory_path).fillna('null')
    keys = list(df['元素'])
    sex = list(df['性别'])
    age = list(df['年龄'])
    inducement = list(df['诱因'])
    time_of_onset = list(df['发病时间'])
    sym_phy = list(df['症状体征'])
    aggravating = list(df['病情加重'])
    pre_treat = list(df['治疗'])
    past_med_his = list(df['既往史'])
    aux_exam = list(df['辅助检查'])
    disease = list(df['疾病'])
    values = []
    for value in zip(sex, age, inducement, time_of_onset, sym_phy, aggravating, pre_treat, past_med_his, aux_exam, disease):
        value = tuple(filter(lambda x: x != 'null', value))
        value = '；'.join(value)
        values.append(value.split('；'))
    content = dict(zip(keys, values))
    json_path = os.path.join('root', 'contradictory.json')
    with open(json_path, 'w') as f:
        content = json.dumps(content, ensure_ascii=False)
        f.write(content)

class Util:
    @staticmethod
    def simple_json_to_excel(json_path, excel_path):
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        keys = list(data.keys())
        values = list(data.values())
        for i, value in enumerate(values):
            values[i] = str(value).replace('[', '').replace(']', '').replace("'", '').replace(", ", '，')
        df = pd.DataFrame()
        df['key'], df['value'] = keys, values
        df.to_excel(excel_path, sheet_name='Sheet1', index=False)

    @staticmethod
    def simple_excel_to_json(excel_path, json_path):
        df = pd.read_excel(excel_path)
        keys = list(df['key'])
        values = list(df['value'])
        for i, value in enumerate(values):
            values[i] = value.split('；')
        content = dict(zip(keys, values))
        with open(json_path, 'w') as f:
            content = json.dumps(content, ensure_ascii=False)
            f.write(content)

    @staticmethod
    def load_json(file_path: str):
        with open(file_path, 'r') as f:
            content = f.readline()
            content = json.loads(content)
        return content

    @staticmethod
    def dump_json(content, file_path: str):
        with open(file_path, 'w') as f:
            content = json.dumps(content, ensure_ascii=False)
            f.write(content)

if __name__ == '__main__':
    save_root_data(os.path.join('root', 'root_mod.xlsx'),
                   os.path.join('root', 'contradictory.xlsx'))