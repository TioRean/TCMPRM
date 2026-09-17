
import numpy as np
import pandas as pd
from itertools import combinations, chain, product
from logger import get_logger
import copy
from itertools import chain
import os
import re
from typing import Union

logger = get_logger('rgenerate')

class RuleGenerator:
    def __init__(self, org_file_path):
        self.tp_group_traits = {'t_color': ('Pale White Tongue', 'Pale Dark Tongue', 'Pale Red Tongue', 'Dark Red Tongue', 'Red Tongue', 'Crimson Tongue', 'Purple Tongue', 'Blue Tongue'),
                                't_nature': ('Old Tongue', 'Tender Tongue', 'null'),
                                't_coating_color': ('White Fur', 'Yellow Fur', 'Grey Fur', 'Black Fur'),
                                't_coating_thickness': ('Scanty Fur', 'Thin Fur', 'Thick Fur'),
                                't_coating_humidity': ('Watery Fur', 'Dry Fur', 'null.1'),
                                't_coating_character': ('Greasy Fur', 'Curdy Fur', 'null.2'),
                                'p_rate': ('Rapid Pulse', 'Swift Pulse', 'Slow Pulse',  'null.3'),
                                'p_rhythm': ('Rapid Intermittent Pulse', 'Irregularly Slow Intermittent Pulse', 'Regularly Intermittent Pulse',  'null.4'),
                                'p_position': ('Floating Pulse', 'Deep Pulse', 'Hidden Pulse', 'null.5'),
                                'p_body': ('Large Pulse', 'Thin Pulse', 'Long Pulse', 'Short Pulse', 'null.6'),
                                'p_strength': ('Deficient Pulse', 'Weak Pulse', 'Faint Pulse', 'Excess Pulse', 'Flickering Pulse', 'Absent Pulse',  'null.7'),
                                'p_fluency': ('Slippery Pulse', 'Rough Pulse', 'null.8'),
                                'p_tension': ('Wiry Pulse', 'Tense Pulse', 'Moderate Pulse', 'Hard Pulse', 'Soft Pulse', 'null.9'),
                                'p_complex': ('Leathery Pulse', 'Firm Pulse', 'Surging Pulse', 'Stirring Pulse', 'Hollow Pulse', 'Floating, Large and Hollow Pulse', 'Soggy Pulse', 'null.10')}
        self.tp_traits_group = self._get_reverse_dict(self.tp_group_traits)
        self.org_file_path = org_file_path
        self.df = pd.read_excel(self.org_file_path, sheet_name='rule', skiprows=1).fillna('null')
        try:
            self.mdf_df = pd.read_excel(self.org_file_path, sheet_name='modification', skiprows=1).fillna('NA')
        except ValueError:
            self.mdf_df = None
        self._prepare_data()

    def _prepare_data(self):
        for i, row in self.df.iterrows():
            for column in row.keys():
                self.df.loc[i, column] = re.sub(r'[ ]*[;；][ ]*', '；', self.df.loc[i, column])
                self.df.loc[i, column] = self.df.loc[i, column].strip(',.;，。 ')
        for i, row in self.df.loc[:, 'Pale White Tongue':'null.10'].iterrows():
            for column in row.keys():
                if 'T' in self.df.loc[i, column]:
                    temp = self.df.loc[i, column].split('；')
                    while len(temp) < 3:
                        temp.append('')
                    self.df.loc[i, column] = '；'.join(temp)
        if self.mdf_df is not None:
            for i, row in self.mdf_df.loc[:, 'Pale White Tongue':'null.10'].iterrows():
                for column in row.keys():
                    self.mdf_df.loc[i, column] = re.sub(r'[ ]*[;；]+[ ]*', '；', self.mdf_df.loc[i, column])
                    if 'T' in self.mdf_df.loc[i, column]:
                        temp = self.mdf_df.loc[i, column].split('；')
                        while len(temp) < 3:
                            temp.append('')
                        self.mdf_df.loc[i, column] = '；'.join(temp)
                    if re.search(r'[Oo][A-Za-z]+', self.mdf_df.loc[i, column]):
                        temp = self.mdf_df.loc[i, column].split('；')
                        temp[1] = temp[1].lower()
                        self.mdf_df.loc[i, column] = '；'.join(temp)
                    self.mdf_df.loc[i, column] = self.mdf_df.loc[i, column].strip(',.:，。： ')
            for i, row in self.mdf_df.loc[:, 'Condition Aggravation Element Library':'Elements to Remove E'].iterrows():
                for column in row.keys():
                    if re.search(r'[Oo][A-Za-z]+', self.mdf_df.loc[i, column]):
                        temp = self.mdf_df.loc[i, column].lower().split('；')
                        self.mdf_df.loc[i, column] = '；'.join(temp)

    @staticmethod
    def _get_reverse_dict(dict_obj: dict[str:list]):
        reverse_dict = dict()
        for k, vs in dict_obj.items():
            for v in vs:
                reverse_dict[v] = k
        return reverse_dict

    @classmethod
    def _get_new_file_path(cls, org_file_path, new_path_name: str):
        org_file_name, suffix = os.path.splitext(os.path.basename(org_file_path))
        new_file_path = os.path.join(os.path.dirname(org_file_path), org_file_name + f'_{new_path_name}' + suffix)
        return new_file_path

    def _generate_modification(self):
        def _get_modifications():
            content = dict()
            for i, row in self.mdf_df.iterrows():
                if row['Prescription No.'] not in content.keys():
                    content[row['Prescription No.']] = dict()
                if row['Prescription Name'] not in content[row['Prescription No.']]:
                    content[row['Prescription No.']]['Prescription Name'] = row['Prescription Name']
                else:
                    assert content[row['Prescription No.']]['Prescription Name'] == row['Prescription Name']
                content[row['Prescription No.']][row['Modification No.']] = row['Combined Prescription': 'null.10']
            return content

        def _build_bracket_tree(text):
            root = {"content": text, "type": "", "children": []}  # 虚拟根节点
            stack = []
            brackets = {')': '(', ']': '[', '}': '{', '>': '<'}  # 右括号到左括号的映射
            open_brackets = set(brackets.values())  # 所有左括号类型

            for i, char in enumerate(text):
                if char in open_brackets:
                    # 创建新节点，初始化类型和内容
                    node = {"content": "", "type": "", "children": []}
                    stack.append((node, i + 1, char))  # 压入节点、起始位置、左括号字符
                elif char in brackets:
                    if not stack:
                        continue
                    # 检查括号类型是否匹配
                    current_node, start_pos, start_char = stack[-1]
                    if brackets[char] != start_char:
                        continue
                    # 弹出栈顶并填充节点
                    stack.pop()
                    current_node["content"] = text[start_pos:i]
                    current_node["type"] = f"{start_char}{char}"  # 标注括号类型，如 "()"
                    # 挂载到父节点
                    if stack:
                        parent_node, _, _ = stack[-1]
                        parent_node["children"].append(current_node)
                    else:
                        root["children"].append(current_node)
            # return root["children"]
            return root

        def _combi(node):
            content = []
            xs = node['content'].replace(', ', ',').split(',')
            if node['type'] == '()':
                for i in range(1, 2):
                    for x in combinations(xs, i):
                        content.append('&'.join(x))
            elif node['type'] == '[]':
                for i in range(1, 3):
                    for x in combinations(xs, i):
                        content.append('&'.join(x))
            elif node['type'] == '{}':
                for i in range(1, 4):
                    for x in combinations(xs, i):
                        content.append('&'.join(x))
            elif node['type'] == '<>':
                for i in range(1, 5):
                    for x in combinations(xs, i):
                        content.append('&'.join(x))
            else:
                raise Exception
            return content

        def _process_modifications(text):
            # "(B001, B002)&B003, B004, B005, ''"
            text = re.sub(r'''[ ]*['"‘“]+[ ]*['"‘“]+[ ]*''', '', text)
            text = re.sub(r'[ ]*&[ ]*', '&', text)
            text = re.sub(r'[ ]*[（(][ ]*', '(', text)
            text = re.sub(r'[ ]*[）)][ ]*', ')', text)
            text = re.sub(r'[,，]+[ ]*', ',', text)
            tree = _build_bracket_tree(text)
            position = {}
            for i, node in enumerate(tree['children']):
                position[f'pos_{i}'] = _combi(node)
                tree['content'] = tree['content'].replace(node['type'][0] + node['content'] + node['type'][1],
                                                          f'pos_{i}')
            tree['content'] = re.sub(r'[,，]+[ ]*]', ',', tree['content']).split(',')
            content = set()
            for section in tree['content']:
                xs = section.split('&')
                for i, x in enumerate(xs):
                    if re.match(r'pos_', x):
                        xs[i] = position[x]
                    elif re.match(r'B\d+', x):
                        xs[i] = [x]
                    elif x == '':
                        xs[i] = ['']
                    else:
                        raise Exception
                for x in product(*xs):
                    content.add('&'.join(set(x)))
            content = [set(xs.split('&')) for xs in content if xs != '']
            executions = []
            # 删除重复的组合
            for xs in content:
                save_flag = True
                for ys in executions:
                    if not xs - ys and not ys - xs:
                        save_flag = False
                        break
                if save_flag:
                    executions.append(xs)
            executions = [list(filter(lambda x: x != '', xs)) for xs in executions]
            # executions = [str(execution).split('&') for execution in executions]
            return executions

        def _del_elem_unit(elem: str, del_mdf: set):
            elem_unit = [set(x.split('/')) for x in elem.split('&')]
            elem_unit = [x - del_mdf for x in elem_unit]
            elem_unit = list(filter(lambda x:x != '' and x != 'null', elem_unit))
            return elem_unit

        def _check_and_del_elems(pool:set, del_mdf:set, mod: str):
            if not pool:
                return pool, False
            result = []
            cont_flag = False
            if mod == 'denote_pool':
                elem_dict = dict()
                for elem in pool:
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
                elem_dict = {k: list(filter(lambda x: x != '' and x != 'null', v)) for k, v in
                             elem_dict.items()}
                elem_dict = {k: v for k, v in elem_dict.items() if len(v) > 0}
                for k, vs in elem_dict.items():
                    maj_elms = [v.replace('<maj>', '') for v in vs if '<maj>' in v]
                    vs = [v.replace('<maj>', '') for v in vs]
                    if len(maj_elms) > 0 and len(maj_elms) < len(vs):
                        elem_dict[k] = {'<maj>': set(maj_elms), '<other>': set(vs) - set(maj_elms)}
                    else:
                        elem_dict[k] = {'<other>': set(vs)}
                elem_dict = {K: {k: [_del_elem_unit(x, del_mdf) for x in v] for k, v in V.items()}
                             for K, V, in elem_dict.items()}

                for K, V in elem_dict.items():
                    if re.search(r'<[cC]>', K):
                        # 全部元素均需满足，不能与del_mdf存在交集
                        if not all(list(chain.from_iterable(chain.from_iterable(V.values())))):
                            cont_flag = True
                    elif re.search(r'<[oO][a-zA-Z]*>', K) and K != '<own>':
                        # 至少存在一个元素，与del_mdf相减后不能为空集
                        if not list(chain.from_iterable(chain.from_iterable(chain.from_iterable(V.values())))):
                            cont_flag = True
                    for k, v in V.items():
                        for elem_unit in v:
                            elem = '&'.join(['/'.join(x) for x in elem_unit if len(x) > 0])
                            if k == '<maj>' and elem:
                                elem = k + elem
                            if any([re.search(r'<[cC]>', K), re.search(r'<[oO][a-zA-Z]*>', K)]) and K != '<own>' and elem:
                                elem = K + elem
                            if elem:
                                result.append(elem)
            elif mod == 'CEP':
                for elem in pool:
                    elem_unit = _del_elem_unit(elem, del_mdf)
                    if not all(elem_unit):
                        cont_flag = True
                    elem = '&'.join(['/'.join(x) for x in elem_unit if len(x) > 0])
                    if elem:
                        result.append(elem)
            elif mod == 'OEP':
                for elem in pool:
                    elem_unit = _del_elem_unit(elem, del_mdf)
                    if not all(elem_unit):
                        continue
                    elem = '&'.join(['/'.join(x) for x in elem_unit if len(x) > 0])
                    if elem:
                        result.append(elem)
                if not result:
                    cont_flag = True
            else:
                raise Exception
            return result, cont_flag

        mdf_dict = _get_modifications()
        org_df, new_df = copy.deepcopy(self.df), copy.deepcopy(self.df)
        idx = 0
        for i, row in org_df.iterrows():
            # 添加原始的方证规则
            new_df.loc[idx] = row
            psc_num = row['Prescription No.']
            psc_name = row['Prescription Name (Head)']
            row_modif = copy.deepcopy(row)

            org_med_dict = {}
            for med_unit in filter(lambda x: x != 'null', row['Medicine 1':'Medicine 50'].tolist()):
                med_unit = med_unit.split('；')
                org_med_dict[med_unit[0]] = [med_unit[1], med_unit[2], med_unit[3]]

            if row['Modification Type A'] != 'null':
                row_modif['Prescription No.'] = row['Prescription No.'] + '-' + row['Modification Type A']
                row_modif['Rule No.'] = row['Rule No.'] + '-' + row['Modification Type A']
            new_df.loc[idx] = copy.deepcopy(row_modif)
            idx += 1
            if row['Modification Type B'] != 'null':
                # 目标 [['B002'], ['B003'], ['B002', 'B003']]
                # B001,B002,B003,B003&B004,B003&B006,B007,(B001,B002,B003),[B001,B002,B003],
                executions = _process_modifications(str(row['Modification Type B']))
                assert mdf_dict[psc_num]['Prescription Name'] == psc_name
                # 获取加减法规则
                for execution in executions:
                    # executions [['B002'], ['B003'], ['B002', 'B003']]
                    # execution ['B002', 'B003']
                    continue_flag = False
                    rule_num_mdf = execution
                    psc_mdf = []
                    medic_mdf = {'Medicines to Add': dict(), 'Medicines to Subtract': list()}
                    cps_mdf = []
                    opt_mdf = {'Sym and Sign Optional Element Pool with null': set()}
                    cls_mdf = set()
                    others_mdf = {'Onset Time Element Library': set(),
                                  'Condition Aggravation Element Library': set(),
                                  'Inducement Element Library': set(),
                                  'Treatment Element Library': set(),
                                  'Disease Element Library': set(),
                                  }
                    del_mdf = []
                    neg_mdf = []
                    tp_mdf = {}
                    for i, mdf_num in enumerate(execution):
                        mdf = mdf_dict[psc_num][mdf_num]
                        # 合方
                        adding_psc = str(mdf['Combined Prescription']).split('&')
                        psc_mdf.extend(list(filter(lambda x: x not in psc_mdf and x != 'NA', adding_psc)))

                        # 加法
                        adding_meds = {adding_med.split('；')[0]: adding_med.split('；')[1:]
                                       for adding_med in str(mdf['Medicines to Add']).split('&')}
                        for adding_med in adding_meds.keys():
                            if adding_med not in medic_mdf['Medicines to Add'] and adding_meds != 'NA':
                                medic_mdf['Medicines to Add'][adding_med] = adding_meds[adding_med]
                            elif adding_med in medic_mdf['Medicines to Add']:
                                # 新增的两种相同药物的加法都应当同时大于原本规则的药物剂量，否则药物均为减量，或一增量一减量，为矛盾
                                if adding_med not in org_med_dict.keys():
                                    if float(adding_meds[adding_med][0]) > float(medic_mdf['Medicines to Add'][adding_med][0]):
                                        medic_mdf['Medicines to Add'][adding_med] = adding_meds[adding_med]
                                else:
                                    if float(adding_meds[adding_med][0]) > float(org_med_dict[adding_med][0]) and float(
                                            medic_mdf['Medicines to Add'][adding_med][0]) > float(org_med_dict[adding_med][0]):
                                        # 若存在两种加法药物剂量不同，按剂量大修改规则
                                        if float(adding_meds[adding_med][0]) > float(medic_mdf['Medicines to Add'][adding_med][0]):
                                            medic_mdf['Medicines to Add'][adding_med] = adding_meds[adding_med]
                                    else:
                                        continue_flag = True
                                        logger.warning(f"Prescription Name'{psc_name} 'Prescription No.'{psc_num}', "
                                                       f"执行加减法编码{execution}加法药物剂量存在矛盾")
                                        break
                            # 减法
                        dropping_meds = str(mdf['Medicines to Subtract']).split('&')

                        medic_mdf['Medicines to Subtract'].extend(list(filter(lambda x: x not in medic_mdf['Medicines to Subtract']
                                                                       and x != 'NA', dropping_meds)))

                        # Elements to Remove
                        del_mdf.extend(chain.from_iterable(
                            [re.sub(r'<[^>]*>', '', elems).split('；')
                             for elems in mdf['Elements to Remove A':'Elements to Remove E'] if 'NA' not in elems]))

                        # 类别
                        if mdf['Category'] != 'NA':
                            cls_mdf.update(mdf['Category'].split('；'))
                        # 发病时间
                        if mdf['Onset Time Element Library'] != 'NA':
                            others_mdf['Onset Time Element Library'].update(mdf['Onset Time Element Library'].split('；'))
                        # 加重因素
                        if mdf['Condition Aggravation Element Library'] != 'NA':
                            others_mdf['Condition Aggravation Element Library'].update(mdf['Condition Aggravation Element Library'].split('；'))
                        # 诱因
                        if mdf['Inducement Element Library'] != 'NA':
                            others_mdf['Inducement Element Library'].update(mdf['Inducement Element Library'].split('；'))
                        # 治疗
                        if mdf['Treatment Element Library'] != 'NA':
                            others_mdf['Treatment Element Library'].update(mdf['Treatment Element Library'].split('；'))

                        # 必选元素库
                        cps_mdf.extend(
                            chain.from_iterable([elements.split('；') for elements in mdf['Sym and Sign Compulsory Element Pool A':'Sym and Sign Compulsory Element Pool C']
                                                 if 'NA' not in elements]))
                        # Sym and Sign Optional Element Pool
                        for optional in mdf['Sym and Sign Optional Element Pool A':'Sym and Sign Optional Element Pool O']:
                            optional = optional.split('；')
                            if 'null' in optional:
                                opt_mdf['Sym and Sign Optional Element Pool with null'].update(optional)
                            elif 'NA' in optional:
                                continue
                            else:
                                opt_mdf[f'Sym and Sign Optional Element Pool {len(opt_mdf.keys())}'] = optional

                        # Disease Element Library
                        if mdf['Disease Element Library'] != 'NA':
                            others_mdf['Disease Element Library'].update(mdf['Disease Element Library'].split('；'))

                        # 阴性元素
                        neg_mdf.extend(chain.from_iterable(
                            [elems.split('；') for elems in mdf['Elements to Remove A':'Elements to Remove E'] if 'NA' not in elems]))
                        # 舌脉元素
                        new_tp_dict = dict()
                        for tp_group_name, tp_traits in self.tp_group_traits.items():
                            new_tp_dict[tp_group_name] = mdf[tp_traits[0]:tp_traits[-1]].to_dict()

                        for trait_group in new_tp_dict.keys():
                            if trait_group not in tp_mdf.keys():
                                for k, v in new_tp_dict[trait_group].items():
                                    if 'T' in v:
                                        new_tp_dict[trait_group][k] = '；'.join([x + f'_{i}' if j == 1 and x != '' else x
                                                                                for j, x in enumerate(v.split('；'))])
                                tp_mdf[trait_group] = new_tp_dict[trait_group]
                            else:
                                for trait in new_tp_dict[trait_group].keys():
                                    if new_tp_dict[trait_group][trait] == 'F':
                                        cnt = 0
                                        for change in tp_mdf[trait_group].values():
                                            if 'T' in change:
                                                # 一个特征组内 多个特征是否有T，若T只有1个，说明必要条件
                                                cnt += 1
                                        if cnt == 1:
                                            # T 判定为必要条件，但与F相矛盾，该规则跳过
                                            if 'T' in tp_mdf[trait_group][trait]:
                                                continue_flag = True
                                                logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                                               f"执行加减法编码{execution}舌脉加减法执行时存在矛盾")
                                                break
                                            else:
                                                tp_mdf[trait_group][trait] = 'F'
                                        else:
                                            tp_mdf[trait_group][trait] = 'F'
                                    elif new_tp_dict[trait_group][trait] == 'NA':
                                        continue
                                    else:
                                        if tp_mdf[trait_group][trait] == 'F':
                                            continue_flag = True
                                            logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                                           f"执行加减法编码{execution}舌脉加减法执行时存在矛盾")
                                            break
                                        if tp_mdf[trait_group][trait] == 'NA':
                                            tp_mdf[trait_group][trait] = '；；'
                                        tp1 = [set(filter(lambda x: x != '', x.split('&')))
                                               for x in tp_mdf[trait_group][trait].split('；')]

                                        tp2 = [set(filter(lambda x: x != '', x.split('&')))
                                               for x in new_tp_dict[trait_group][trait].split('；')]
                                        tp1[0].update(tp2[0])
                                        tp1[1].update([x + f'_{i}' for x in tp2[1]])
                                        tp2_nes = []
                                        for x in tp2[2]:
                                            temp = []
                                            for y in x.split('/'):
                                                if 'F' in tp_mdf[self.tp_traits_group[y]][y]:
                                                    continue
                                                temp.append(y)
                                            if temp:
                                                tp2_nes.append('/'.join(temp))
                                            else:
                                                continue_flag = True
                                                logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                                               f"执行加减法编码{execution}舌脉加减法执行时存在矛盾")
                                        tp1[2].update(tp2_nes)
                                        tp_mdf[trait_group][trait] = '；'.join(['&'.join(xs) for xs in tp1])
                    if continue_flag:
                        continue

                    # 判断Elements to Remove 与元素加法是否有重叠
                    if del_mdf:
                        _, cps_continue_flag = _check_and_del_elems(set(cps_mdf), set(del_mdf),'CEP')
                        if cps_continue_flag:
                            continue_flag = True
                            logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                           f"执行加减法编码{execution}时因必要元素加减与Elements to Remove 矛盾，予跳过")
                        for column in opt_mdf.keys():
                            _, opt_continue_flag = _check_and_del_elems(set(opt_mdf[column]), set(del_mdf),
                                                                           'OEP')
                            if opt_continue_flag:
                                continue_flag = True
                                logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                               f"执行加减法编码{execution}时因备选元素与Elements to Remove 矛盾，予跳过")
                        for column in others_mdf.keys():
                            _, other_continue_flag = _check_and_del_elems(set(others_mdf[column]),
                                                                              set(del_mdf),
                                                                              'denote_pool')
                            if other_continue_flag:
                                continue_flag = True
                                logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                               f"执行加减法编码{execution}时删除了带<C>的任一元素或<OA><OB>标识的所有元素元素，予跳过")

                        _, cls_continue_flag = _check_and_del_elems(set(cls_mdf), set(del_mdf),
                                                                       'OEP')
                        if cls_continue_flag:
                            continue_flag = True
                            logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                           f"执行加减法编码{execution}时因类型元素与Elements to Remove 矛盾，予跳过")
                        if continue_flag:
                            continue


                    # 药物加法与减法重叠判定：
                    if len(set(medic_mdf['Medicines to Add'].keys()) & set(medic_mdf['Medicines to Subtract'])) >= 1:
                        continue_flag = True
                        logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                       f"执行加减法编码{execution}时药物加法与减法重叠")
                    if continue_flag:
                        continue

                    if len(opt_mdf['Sym and Sign Optional Element Pool with null']) >= 1:
                        opt_mdf['Sym and Sign Optional Element Pool with null'] = list(opt_mdf['Sym and Sign Optional Element Pool with null'])
                    else:
                        opt_mdf.pop('Sym and Sign Optional Element Pool with null')

                    for trait_group in tp_mdf.keys():
                        tp_mdf[trait_group] = {trait: tp_mdf[trait_group][trait] for trait in tp_mdf[trait_group]
                                               if tp_mdf[trait_group][trait] != 'NA'}

                    # 根据加减法产生新的规则
                    # Prescription No.
                    new_psc_num_dict = {'Prescription No.': copy.deepcopy(row_modif['Prescription No.']) + '-' + '&'.join(rule_num_mdf)}
                    # 添加新的Rule No.
                    new_rule_num_dict = {
                        'Rule No.': copy.deepcopy(row_modif['Rule No.']) + '-' + '&'.join(rule_num_mdf)}

                    new_cps_dict = copy.deepcopy(row_modif['Sym and Sign Compulsory Element Pool A':'Sym and Sign Compulsory Element Pool J'].to_dict())

                    for column in new_cps_dict.keys():
                        if new_cps_dict[column] == 'null':
                            new_cps_dict[column] = []
                        else:
                            new_cps_dict[column] = new_cps_dict[column].split('；')

                    for column in new_cps_dict.keys():
                        if not len(new_cps_dict[column]) > 0:
                            new_cps_dict[column] = cps_mdf
                            break
                    else:
                        logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                       f"执行加减法编码{execution}时Sym and Sign Compulsory Element Pool 数量不足，予跳过")

                    # 添加Sym and Sign Optional Element Pool 的加减法
                    # 新添加的元素若与原规则的元素重合，不对重复的元素做处理，将来规则生成时，重复的元素会自动删为1个
                    new_opt_dict = copy.deepcopy(row_modif['Sym and Sign Optional Element Pool A':'Sym and Sign Optional Element Pool AD'].to_dict())
                    for column in new_opt_dict.keys():
                        if new_opt_dict[column] == 'null':
                            new_opt_dict[column] = []
                        else:
                            new_opt_dict[column] = new_opt_dict[column].split('；')
                    # 添加加减法
                    for new_opt_elems in opt_mdf.values():
                        for column in new_opt_dict.keys():
                            if len(new_opt_dict[column]) > 0:
                                continue
                            else:
                                new_opt_dict[column] = new_opt_elems
                                break
                    # 添加发病时间/诱因/治疗/Disease Element Library的加减法
                    new_cls_dict =  {'Category':copy.deepcopy(row_modif['Category'])}
                    for x in new_cls_dict.keys():
                        if new_cls_dict[x] == 'null':
                            new_cls_dict[x] = []
                        else:
                            new_cls_dict[x] = new_cls_dict[x].split('；')
                        new_cls_dict[x].extend(cls_mdf)
                        new_cls_dict[x] = list(set(new_cls_dict[x]))

                    new_others_dict = {
                        'Onset Time Element Library': copy.deepcopy(row_modif['Onset Time Element Library']),
                        'Condition Aggravation Element Library': copy.deepcopy(row_modif['Condition Aggravation Element Library']),
                        'Inducement Element Library': copy.deepcopy(row_modif['Inducement Element Library']),
                        'Treatment Element Library': copy.deepcopy(row_modif['Treatment Element Library']),
                        'Disease Element Library': copy.deepcopy(row_modif['Disease Element Library']),
                    }

                    for x in new_others_dict.keys():
                        if new_others_dict[x] == 'null':
                            new_others_dict[x] = []
                        else:
                            new_others_dict[x] = new_others_dict[x].split('；')
                        new_others_dict[x].extend(others_mdf[x])
                        new_others_dict[x] = list(set(new_others_dict[x]))

                    # 按加减法删除规则部分元素
                    if del_mdf:
                        for column in new_cps_dict.keys():
                            new_cps_dict[column], _ = _check_and_del_elems(set(new_cps_dict[column]), set(del_mdf), 'CEP')

                        for column in new_opt_dict.keys():
                            new_opt_dict[column], _ = _check_and_del_elems(set(new_opt_dict[column]), set(del_mdf),
                                                                                      'OEP')

                        for column in new_others_dict.keys():
                            new_others_dict[column], _ = _check_and_del_elems(set(new_others_dict[column]),
                                                                                           set(del_mdf),
                                                                                           'denote_pool')

                        for column in new_cls_dict.keys():
                            new_cls_dict[column], _ = _check_and_del_elems(set(new_cls_dict[column]), set(del_mdf),
                                                                                      'OEP')

                    # 修改舌脉元素库加减法
                    new_tp_dict = copy.deepcopy(row_modif['Pale White Tongue':'null.10'].to_dict())
                    for trait_group in tp_mdf.keys():
                        for trait in tp_mdf[trait_group]:
                            if tp_mdf[trait_group][trait] == 'F':
                                new_tp_dict[trait] = 'null'
                            else:
                                if new_tp_dict[trait] == 'F':
                                    continue_flag = True
                                    logger.warn(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                                f"执行加减法编码{execution}舌脉元素库加减法执行时存在矛盾")
                                    break
                                if new_tp_dict[trait] == 'null':
                                    new_tp_dict[trait] = '；；'

                                tp1 = [set(filter(lambda x: x != '', x.split('&')))
                                       for x in new_tp_dict[trait].split('；')]
                                tp2 = [set(filter(lambda x: x != '', x.split('&')))
                                       for x in tp_mdf[trait_group][trait].split('；')]
                                tp2_nes = []
                                for x in tp2[2]:
                                    temp = []
                                    for y in x.split('/'):
                                        if 'F' in new_tp_dict[y]:
                                            continue
                                        temp.append(y)
                                    if temp:
                                        tp2_nes.append('/'.join(temp))
                                    else:
                                        continue_flag = True
                                        logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                                       f"执行加减法编码{execution}舌脉加减法执行时存在矛盾")
                                tp1[0].update(tp2[0])
                                tp1[1].update(tp2[1])
                                tp1[2].update(tp2_nes)
                                new_tp_dict[trait] = '；'.join(['&'.join(xs) for xs in tp1])

                    # 合方添加
                    new_psc_dict = {'Prescription Name': copy.deepcopy(row_modif['Prescription Name']).split('；')}
                    for adding_psc in psc_mdf:
                        if adding_psc not in new_psc_dict['Prescription Name']:
                            new_psc_dict['Prescription Name'].append(adding_psc)
                    # 药物加减法
                    new_medic_dict = copy.deepcopy(row_modif['Medicine 1':'Medicine 50'].to_dict())
                    new_medic_dict = {x: y.split('；') for x, y in new_medic_dict.items()}
                    new_medics = {y[0]: [y[1:], x] for x, y in new_medic_dict.items()}
                    for adding_med in medic_mdf['Medicines to Add']:
                        if adding_med in new_medics.keys():
                            # 如果加法药物与原规则药物重合，则覆盖之（用于调整药物剂量）
                            # 确保药物剂量单位相同，否则跳过
                            if new_medic_dict[new_medics[adding_med][1]][2] != medic_mdf['Medicines to Add'][adding_med][1]:
                                continue_flag = True
                                logger.warning(f"Prescription Name'{psc_name}' Prescription No.'{psc_num}', "
                                               f"执行加减法编码{execution}药物加减法中剂量单位不统一")
                                continue
                            new_medic_dict[new_medics[adding_med][1]][1] = medic_mdf['Medicines to Add'][adding_med][0]
                            continue
                        for num in new_medic_dict.keys():
                            if 'null' not in new_medic_dict[num]:
                                continue
                            else:
                                new_medic_dict[num] = [adding_med] + medic_mdf['Medicines to Add'][adding_med]
                                break
                    for dropping_med in medic_mdf['Medicines to Subtract']:
                        for num in new_medic_dict.keys():
                            if 'null' in new_medic_dict[num]:
                                continue
                            if new_medic_dict[num][0] == dropping_med:
                                new_medic_dict[num] = ['null']
                    new_medic_dict = {x: '；'.join(y) for x, y in new_medic_dict.items()}
                    if continue_flag:
                        continue
                    pattern = copy.deepcopy(row_modif)
                    merged_dict_A = {**new_cps_dict, **new_opt_dict, **new_cls_dict, **new_others_dict,
                                     **new_psc_dict}
                    merged_dict_B = {**new_psc_num_dict, **new_rule_num_dict, **new_tp_dict, **new_medic_dict}
                    for column in merged_dict_A.keys():
                        if len(merged_dict_A[column]) == 0:
                            pattern[column] = None
                        else:
                            pattern[column] = '；'.join(merged_dict_A[column])
                    for column in merged_dict_B.keys():
                        pattern[column] = merged_dict_B[column]

                    new_df.loc[idx] = pattern
                    idx += 1
        new_df.drop('Modification Type A', axis=1, inplace=True)
        new_df.drop('Modification Type B', axis=1, inplace=True)
        return new_df

    def generate_rules(self, save_path: str, apply_modification: bool = True, shuffle: bool = True):
        if apply_modification:
            self.df = self._generate_modification()
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.fillna('')
        self.df.replace('null', np.NaN, inplace=True)
        self.df.to_excel(save_path, sheet_name='rule', index=False)

def parse_root_elems_list(TCMRule_path:str, save_path):

    def replace_tag(text: str, n: int=None) -> str:
        pattern = r'(<c?[cCoOaA]z?)(\d+_?\d*)(>)'
        def repl_func(match):
            prefix = match.group(1)
            suffix = match.group(3)
            new_tag = prefix.lower() + str(n) + suffix
            return new_tag
        return re.sub(pattern, repl_func, text)

    def transfer_tag_list(text_list):
        pattern = re.compile(r'<([Cc]\d+|[AaOo]z?\d+)>(.*)')
        res_dict = {}
        for text in text_list:
            match = pattern.match(text)
            if not match:
                continue  # 无标签直接跳过
            full_tag = match.group(0).split('>')[0] + '>'  # 完整标签如<o1>
            content = match.group(2).strip()
            if not content:
                continue
            if full_tag not in res_dict:
                res_dict[full_tag] = []
            res_dict[full_tag].append(content)
        res = [k + '{' + '；'.join(v) +'}' for k, v in res_dict.items()]
        res = '；'.join(res)
        return res

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
                    denote = match.group(1)  # 括号外的内容
                    elements = match.group(2).split('；')  # 括号内的内容
                    new_unit = '；'.join([denote + element for element in elements])
                    content = content.replace(unit, new_unit)
            content = content.split('；')
            return content
    def get_elements_list_for_tongue_pulse(dataframe, i):
        trait_groups = (
            ('Pale White Tongue', 'Blue Tongue'), ('Old Tongue', 'null'), ('White Fur', 'Black Fur'), ('Scanty Fur', 'Thick Fur'), ('Watery Fur', 'null.1'),
            ('Greasy Fur', 'null.2'), ('Rapid Pulse', 'null.3'), ('Rapid Intermittent Pulse', 'null.4'), ('Floating Pulse', 'null.5'), ('Large Pulse', 'null.6'),
            ('Deficient Pulse', 'null.7'), ('Slippery Pulse', 'null.8'), ('Wiry Pulse', 'null.9'), ('Leathery Pulse', 'null.10'))
        content = []
        for trait_group in trait_groups:
            traits = get_elements(dataframe.loc[i, trait_group[0]:trait_group[1]])
            for k, trait in enumerate(traits):
                while len(trait) < 3:
                    trait.append('')
                traits[k] = '；'.join(trait)
            content.append(traits)
        return content

    def _tong_pul_to_dict(t_color, t_nature, t_coating_thickness, t_coating_color, t_coating_humidity,
                          t_coating_character, p_rate, p_rhythm, p_position, p_body, p_strength, p_fluency,
                          p_tension, p_complex):
        result = {'t_color': t_color, 't_nature': t_nature, 't_coating_thickness': t_coating_thickness,
                  't_coating_color': t_coating_color, 't_coating_humidity': t_coating_humidity,
                  't_coating_character': t_coating_character, 'p_rate': p_rate, 'p_rhythm': p_rhythm,
                  'p_position': p_position, 'p_body': p_body, 'p_strength': p_strength,
                  'p_fluency': p_fluency, 'p_tension': p_tension, 'p_complex': p_complex}
        return result

    def _prepare_tongue_pulse(tongue_pulses, lib_num):

        tong_pul_en_ch_map = {
            'Pale White Tongue': '舌淡白', 'Pale Dark Tongue': '舌淡暗', 'Pale Red Tongue': '舌淡红',
            'Dark Red Tongue': '舌暗红', 'Red Tongue': '舌红', 'Crimson Tongue': '舌绛', 'Purple Tongue':
                '舌紫', 'Blue Tongue': '舌青', 'Old Tongue': '舌老', 'Tender Tongue': '舌嫩', 'White Fur': '白苔',
            'Yellow Fur': '黄苔', 'Grey Fur': '灰苔', 'Black Fur': '黑苔', 'Scanty Fur': '苔少', 'Thin Fur': '苔薄',
            'Thick Fur': '苔厚', 'Watery Fur': '苔水滑', 'Dry Fur': '苔燥',
            'Greasy Fur': '苔腻', 'Curdy Fur': '苔腐', 'Rapid Pulse': '脉数', 'Swift Pulse': '脉疾',
            'Slow Pulse': '脉迟',
            'Rapid Intermittent Pulse': '脉促', 'Irregularly Slow Intermittent Pulse': '脉结',
            'Regularly Intermittent Pulse': '脉代', 'Floating Pulse': '脉浮', 'Deep Pulse': '脉沉',
            'Hidden Pulse': '脉伏',
            'Large Pulse': '脉大', 'Thin Pulse': '脉细', 'Long Pulse': '脉长', 'Short Pulse': '脉短',
            'Deficient Pulse': '脉虚', 'Weak Pulse': '脉弱', 'Faint Pulse': '脉微', 'Excess Pulse': '脉实',
            'Flickering Pulse': '脉弹指', 'Absent Pulse': '无脉', 'Slippery Pulse': '脉滑', 'Rough Pulse': '脉涩',
            'Wiry Pulse': '脉弦', 'Tense Pulse': '脉紧', 'Moderate Pulse': '脉缓', 'Hard Pulse': '脉硬',
            'Soft Pulse': '脉软', 'Leathery Pulse': '革脉', 'Firm Pulse': '牢脉', 'Surging Pulse': '洪脉',
            'Stirring Pulse': '动脉', 'Hollow Pulse': '芤脉', 'Floating, Large and Hollow Pulse': '浮大中空脉',
            'Soggy Pulse': '濡脉', 'null': 'null'
        }
        tong_pul = _tong_pul_to_dict(*tongue_pulses)
        tps = dict()
        # neg_set = set()
        num_dict = dict()
        for i, group_name in enumerate(tong_pul.keys()):
            tps[group_name] = set()
            for data in tong_pul[group_name]:
                data_split = data.replace('; ', '；').replace(';', '；').split('；')
                if data_split[0] in tong_pul_en_ch_map:
                    data_split[0] = tong_pul_en_ch_map[data_split[0]]
                depend_elems = data_split[2].split('&')
                if '' in depend_elems:
                    depend_elems.remove('')
                if depend_elems:
                    data_split[0] = data_split[0] + '&' + '&'.join(depend_elems)
                if data_split[1]:
                    for cross_lib in data_split[1].split('&'):
                        cross_lib = f'<{cross_lib}>'
                        if cross_lib not in num_dict.keys():
                            num_dict[cross_lib] = len(num_dict)
                        cross_lib = replace_tag(cross_lib, num_dict[cross_lib])
                        data_split[0] = cross_lib + data_split[0]
                tps[group_name].add(data_split[0])

        result = dict()
        for x in tps.keys():
            if 'null' in tps[x]:
                tps[x] = list(filter(lambda z: z != 'null', tps[x]))
                if tps[x]:
                    result[x] = f'<az{lib_num}>' +'{' + '；'.join(tps[x]) + '}'
                    lib_num += 1
            else:
                if tps[x]:
                    result[x] = f'<a{lib_num}>' + '{' + '；'.join(tps[x]) + '}'
                    lib_num += 1
        return result, lib_num

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

    df = pd.read_excel(TCMRule_path)
    df = df.fillna('None')
    df_2 = df.loc[:, 'Pale White Tongue':'null.10'].astype(str)
    for column in list(df_2.columns):
        if re.match(r'null\.\d+', column):
            df_2.loc[:, column] = df_2[column].str.replace('T', 'null', regex=False)
        else:
            df_2.loc[:, column] = df_2[column].str.replace('F', 'null', regex=False)
            df_2.loc[:, column] = df_2[column].str.replace('T', column, regex=False)
    new_col = ['t_color', 't_nature', 't_coating_thickness',
                  't_coating_color', 't_coating_humidity',
                  't_coating_character', 'p_rate', 'p_rhythm',
                  'p_position', 'p_body', 'p_strength',
                  'p_fluency', 'p_tension', 'p_complex']
    for col_name in new_col:
        if col_name not in df.columns:
            df[col_name] = ""
    for i, row in df.iterrows():
        lib_num = 0
        classification = get_elements(row['Category'])
        for k, x in enumerate(classification):
            if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[oOaA]\d+>', x):
                classification[k] = replace_tag(x, lib_num)
            else:
                classification[k] = f'<a{lib_num}>' + x
        classification = transfer_tag_list(classification)
        df.loc[i, 'Category'] = classification
        lib_num += 1

        sym_sign_container = []
        start_idx = df.columns.get_loc('Sym and Sign Compulsory Element Pool A')
        end_idx = df.columns.get_loc('Sym and Sign Compulsory Element Pool J')
        target_cols = df.columns[start_idx: end_idx + 1]
        for n, col in enumerate(target_cols):
            if row[col] == 'None':
                continue
            compulsory = f'<c{lib_num}>' + '{' + row[col] + '}'
            sym_sign_container.append(compulsory)
            lib_num += 1

        start_idx = df.columns.get_loc('Sym and Sign Optional Element Pool A')
        end_idx = df.columns.get_loc('Sym and Sign Optional Element Pool AD')
        target_cols = df.columns[start_idx: end_idx + 1]
        for n, col in enumerate(target_cols):
            if row[col] == 'None':
                continue
            optional = get_elements(row[col])
            if 'null' in optional:
                optional = list(filter(lambda z: z !='null', optional))
                optional = '；'.join(optional)
                optional = f'<oz{lib_num}>' + '{' + optional + '}'
            else:
                optional = '；'.join(optional)
                optional = f'<o{lib_num}>' + '{' + optional + '}'
            # df.loc[i, col] = optional
            sym_sign_container.append(optional)
            lib_num += 1
        df.loc[i, 'Sym and Sign Pool'] = '；'.join(sym_sign_container)

        sex = get_elements(row['Gender'])
        if not sex:
            df.loc[i, 'Gender'] = f'<az{lib_num}>{{男；女}}'
        else:
            for k, x in enumerate(sex):
                if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[oOaA]\d+>',x):
                    sex[k] = replace_tag(x, lib_num)
                else:
                    sex[k] = f'<az{lib_num}>' + x
            sex = transfer_tag_list(sex)
            df.loc[i, 'Gender'] = sex
        lib_num += 1

        age = get_elements(df.loc[i, 'Age'])
        if not age:
            df.loc[i, 'Age'] = f'<az{lib_num}>{{成年人：18-岁->65-岁}}'
        else:
            for k, x in enumerate(age):
                if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[oOaA]\d+>',x):
                    age[k] = replace_tag(x, lib_num)
                else:
                    age[k] = f'<az{lib_num}>' + x
            age = transfer_tag_list(age)
            df.loc[i, 'Age'] = age
        lib_num += 1

        time_of_onset = get_elements(df.loc[i, 'Onset Time Element Library'])
        if not time_of_onset:
            df.loc[i, 'Onset Time Element Library'] = f'<az{lib_num}>{{成年人：18-岁->65-岁}}'
        else:
            for k, x in enumerate(time_of_onset):
                if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[oOaA]\d+>', x):
                    time_of_onset[k] = replace_tag(x, lib_num)
                else:
                    time_of_onset[k] = f'<az{lib_num}>' + x
            time_of_onset = transfer_tag_list(time_of_onset)
            df.loc[i, 'Onset Time Element Library'] = time_of_onset
        lib_num += 1

        aggravating = get_elements(df.loc[i, 'Condition Aggravation Element Library'])
        for k, x in enumerate(aggravating):
            if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[oOaA]\d+>', x):
                aggravating[k] = replace_tag(x, lib_num)
            else:
                aggravating[k] = f'<az{lib_num}>' + x
        aggravating = transfer_tag_list(aggravating)
        df.loc[i, 'Condition Aggravation Element Library'] = aggravating
        lib_num += 1

        inducement = get_elements(df.loc[i, 'Inducement Element Library'])
        for k, x in enumerate(inducement):
            if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[aAoO]\d+>', x):
                inducement[k] = replace_tag(x, lib_num)
            else:
                inducement[k] = f'<az{lib_num}>' + x
        inducement = transfer_tag_list(inducement)
        df.loc[i, 'Inducement Element Library'] = inducement
        lib_num += 1

        previous_treatment = get_elements(df.loc[i, 'Treatment Element Library'])
        for k, x in enumerate(previous_treatment):
            if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[aAoO]\d+>', x):
                previous_treatment[k] = replace_tag(x, lib_num)
            else:
                previous_treatment[k] = f'<az{lib_num}>' + x
        previous_treatment = transfer_tag_list(previous_treatment)
        df.loc[i, 'Treatment Element Library'] = previous_treatment
        lib_num += 1

        auxiliary_examination = get_elements(df.loc[i, 'Auxiliary Examination Element Library'])
        for k, x in enumerate(auxiliary_examination):
            if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[aAoO]\d+>', x) :
                auxiliary_examination[k] = replace_tag(x, lib_num)
            else:
                auxiliary_examination[k] = f'<az{lib_num}>' + x
        auxiliary_examination = transfer_tag_list(auxiliary_examination)
        df.loc[i, 'Auxiliary Examination Element Library'] = auxiliary_examination
        lib_num += 1

        diagnosis = get_elements(df.loc[i, 'Disease Element Library'])
        for k, x in enumerate(diagnosis):
            if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[oOaA]\d+>', x):
                diagnosis[k] = replace_tag(x, lib_num)
            else:
                diagnosis[k] = f'<az{lib_num}>' + x
        diagnosis = transfer_tag_list(diagnosis)
        df.loc[i, 'Disease Element Library'] = diagnosis
        lib_num += 1

        past_med_history = get_elements(df.loc[i, 'Past History Element Library'])
        for k, x in enumerate(past_med_history):
            if has_matching_string(r'<[cC]\d+>', x) or has_matching_string(r'<[aAoO]\d+>', x):
                past_med_history[k] = replace_tag(x, lib_num)
            else:
                past_med_history[k] = f'<az{lib_num}>' + x
        past_med_history = transfer_tag_list(past_med_history)
        df.loc[i, 'Past History Element Library'] = past_med_history
        lib_num += 1

        tongue_pulse = get_elements_list_for_tongue_pulse(dataframe=df_2, i=i)
        tps, lib_num = _prepare_tongue_pulse(tongue_pulse, lib_num)
        for col, val in tps.items():
            df.loc[i, col] = val

        df.loc[i, 'lib_num'] = lib_num

    start_col = "Pale White Tongue"
    end_col = "null.10"
    s_idx = df.columns.get_loc(start_col)
    e_idx = df.columns.get_loc(end_col)
    df = df.drop(columns=df.columns[s_idx: e_idx + 1])

    start_col = "Sym and Sign Compulsory Element Pool A"
    end_col = "Sym and Sign Optional Element Pool AD"
    s_idx = df.columns.get_loc(start_col)
    e_idx = df.columns.get_loc(end_col)
    df = df.drop(columns=df.columns[s_idx: e_idx + 1])
    df.fillna('')
    df.replace('None', np.NaN, inplace=True)
    df.to_excel(save_path, sheet_name='rule', index=False)

if __name__ == '__main__':
    org_file_path = os.path.join('original_tcmdtr.xlsx')
    mod_rule_path = os.path.join('original_tcmdtr_mod.xlsx')
    save_path = os.path.join('tcmdtr.xlsx')

    rg = RuleGenerator(org_file_path=org_file_path)
    rg.generate_rules(mod_rule_path, apply_modification=True, shuffle=True)
    parse_root_elems_list(mod_rule_path, save_path)

