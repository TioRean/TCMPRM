
import math
import random
import re
from itertools import chain, islice, tee
import copy
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
import numpy as np
from typing import Sequence, Union, Literal
from logger import get_logger
import jsonlines
import os
import json
import torch
from mingzi import mingzi
from modelscope import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import tempfile


os.environ['VLLM_USE_MODELSCOPE'] = 'true'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
torch.cuda.set_device(0)
device = "cuda"
logger = get_logger('cgenerate')

input_template_1 = {
    'head': '结构化信息如下：',
    'task': {
        'task_prompts': ['将下列结构化信息改写成用自然语言书写的中医医案', ]
    },
    'requirements': {
        'basic_requirement_prompts': [
            '对结构化信息进行同义词转换后，改写成自然语言',
            '生成内容只涉及病史信息，禁止生成关于中医病因病机、辨证分析、四诊分析、辨证结果、诊疗计划、处方、方剂组成、煎服法、注意事项的内容',
            '不改变原信息含义',
            '文字避免不必要的重复',
            '书写用语符合医学规范，逻辑清晰',
            '书写内容及各个时间节点符合逻辑，符合现实，避免前后文自相矛盾',
        ],
        'style_prompts': [
            '将结构化信息改写为中医医案，医案按照主诉、现病史、刻下、查体、既往史、个人史、辅助检查的标准顺序格式书写',
            '女性患者需额外写月经史、婚育史',
            '不必保留原结构化信息的格式'],
        'chief_complaint_prompts': ['确保主诉精简规范，主诉症状或体征合计不超过3个',
                                    '主诉不超过20个字',
                                    '若未给定主诉，则从临床表现中选择最急需解决的症状或体征作为主诉，主诉未必是前几个临床表现',
                                    '若临床表现涉及多系统疾病，选择单个系统的疾病作为主要诊断，主要诊断与主诉相对应',
                                    '虽然中医信息常涉及多个系统的症状体征，但是主诉书写时禁止对应多个不相关的系统或西医诊断，而应对应单个系统或单个西医诊断，其余信息无需在主诉中体现',
                                    ],
        'sym_phy_prompts': [
            '结构化信息中临床表现为乱序，需以主诉为核心调整书写顺序，主次分明，相关的临床表现相邻',
            '参考病史信息和对应处方添加指定的额外信息，但添加的额外信息不影响处方结果',
            ['适当添加与当前辨证论治无关的症状体征信息，不得额外增加支持当前辨证论治结果的其他症状体征',
             ''],
            '按病例书写习惯在刻下症一栏中添加适当的阴性信息',
            '若未指定一般情况、进食、睡眠、大便、小便、查体，需按病例书写习惯补充相关阴性信息',
            '若个别临床表现与整体明显不符，可删去',
            '若舌象或脉象信息存在矛盾，需考虑不同部位存在不同舌象脉象的可能'],
        'pre_treat_prompts': ['可按病例书写习惯适当添加既往治疗的中西药物', ],
        'aux_exam_prompts': ['可按病例书写习惯生成额外的支持诊断或鉴别诊断的辅助检查内容', ],
        'past_med_his_prompts': [
            '若结构化信息中未指定既往史、个人史、婚育史、月经史，需按病例书写习惯补充相关阴性信息',
            '如有必要，可按病例书写习惯适当添加有利于诊断或鉴别诊断的既往史、个人史、婚育史、月经史',
            [{'prompt': '患者既往有<space_0>相关基础病，<space_1>手术病史，随机为其添加相关既往病史',
              'space': [
                  [['循环系统', '呼吸系统', '消化系统', '神经系统', '运动系统',
                    '泌尿生殖系统'], (1, 2)],
                  [['无', '1种', '2种'], 1],
              ]}, '', '', '', ]],
        'diagnosis_prompts': ['若未提供诊断信息，则不书写诊断结果部分',
                              ],
        'explanation_prompts': ['禁止添加关于本病例文本如何被编写的“元说明”和自我评价',
                                '生成内容中禁止提到任何上述文本生成要求的内容']
    }

}

input_templates_2 = {
    'head': '结构化信息如下：',
    'task': {
        'task_prompts': ['将下列结构化信息改写成用自然语言书写的中医医案', ]
    },
    'requirements': {
        'basic_requirement_prompts': [
            '对结构化信息进行同义词转换后，改写成自然语言',
            '生成内容只涉及病史信息，禁止生成关于中医病因病机、辨证分析、四诊分析、辨证结果、诊疗计划、处方、方剂组成、煎服法、注意事项的内容',
            '不改变原信息含义',
            '文字避免不必要的重复',
            '书写用语符合医学规范，逻辑清晰',
            '书写内容及各个时间节点符合逻辑，符合现实，避免前后文自相矛盾',
        ],
        'style_prompts': [
            '模仿近现代中医师的中医医案格式及语言风格书写，不必保留原结构化信息的格式，不需按照主诉、现病史、既往史等的顺序书写'],
        'chief_complaint_prompts': ['确保主诉精简规范，主诉症状或体征合计不超过3个',
                                    '主诉不超过20个字',
                                    '若未给定主诉，则从临床表现中选择最急需解决的症状或体征作为主诉，主诉未必是前几个临床表现',
                                    '若临床表现涉及多系统疾病，选择单个系统的疾病作为主要诊断，主要诊断与主诉相对应',
                                    '虽然中医信息常涉及多个系统的症状体征，但是主诉书写时禁止对应多个不相关的系统或西医诊断，而应对应单个系统或单个西医诊断，其余信息无需在主诉中体现',
                                    ],
        'sym_phy_prompts': [
            '结构化信息中临床表现为乱序，需以主诉为核心调整书写顺序，主次分明，相关的临床表现相邻'
            '参考病史信息和对应处方添加指定的额外信息，但添加的额外信息不影响处方结果',
            [
                '适当添加与当前辨证论治无关的症状体征信息，但不影响最终辨证结果，不得额外增加支持当前辨证论治结果的其他症状体征',
                ''],
            '按病例书写习惯在刻下症一栏中添加适当的阴性信息',
            '若未指定一般情况、进食、睡眠、大便、小便、查体，需按病例书写习惯补充相关阴性信息',
            '若个别临床表现与整体明显不符，可删去',
            '若舌象或脉象信息存在矛盾，需考虑不同部位存在不同舌象脉象的可能'],
        'pre_treat_prompts': ['可适当添加既往治疗的中西药物', ],
        'aux_exam_prompts': ['可生成额外的支持诊断或鉴别诊断的辅助检查内容', ],
        'past_med_his_prompts': [
            '如有必要，可适当添加有利于诊断或鉴别诊断的既往史、个人史、婚育史、月经史',
            [{'prompt': '患者既往有<space_0>相关基础病，<space_1>手术病史，随机为其添加相关既往病史',
              'space': [
                  [['循环系统', '呼吸系统', '消化系统', '神经系统', '运动系统',
                    '泌尿生殖系统'], (1, 2)],
                  [['无', '1种', '2种'], 1],
              ]}, '', '', '', ]],
        'diagnosis_prompts': ['若未提供诊断信息，则不书写诊断结果部分', ],
    }

}

input_templates_3 = {
    'head': '结构化信息如下：',
    'task': {
        'task_prompts': ['将下列结构化信息改写成用自然语言书写的中医医案', ]
    },
    'requirements': {
        'basic_requirement_prompts': [
            '对结构化信息进行同义词转换后，改写成自然语言',
            '生成内容只涉及病史信息，禁止生成关于中医病因病机分析、辨证分析、四诊分析、辨证结果、诊疗计划、处方、方剂组成、煎服法、注意事项的内容',
            '不改变原信息含义',
            '文字避免不必要的重复',
            '书写内容及各个时间节点符合逻辑，符合现实，避免前后文自相矛盾',
        ],
        'style_prompts': [
            '将结构化信息改为口语化的中医医案',
            '不必保留原结构化信息的格式',
        ],
        'chief_complaint_prompts': [
            '若未给定主诉，则从临床表现中选择最急需解决的症状或体征作为主诉，主诉未必是前几个临床表现',
            '若临床表现涉及多系统疾病，选择单个系统的疾病作为主要诊断，主要诊断与主诉相对应',
            '虽然中医信息常涉及多个系统的症状体征，但是主诉书写时禁止对应多个不相关的系统或西医诊断，而应对应单个系统或单个西医诊断，其余信息无需在主诉中体现',
        ],
        'sym_phy_prompts': [
            '结构化信息中临床表现为乱序，需以主诉为核心调整书写顺序，主次分明，相关的临床表现相邻',
            '参考病史信息和对应处方添加指定的额外信息，但添加的额外信息不影响处方结果',
            [
                '适当添加与当前辨证论治无关的症状体征信息，但不影响最终辨证结果，不得额外增加支持当前辨证论治结果的其他症状体征',
                ''],
            '按病例书写习惯在刻下症处添加适当的阴性信息',
            '若未指定一般情况、进食、睡眠、大便、小便、查体，需按病例书写习惯补充相关阴性信息',
            '若个别临床表现与整体明显不符，可删去',
            '若舌象或脉象信息存在矛盾，需考虑不同部位存在不同舌象脉象的可能'],
        'pre_treat_prompts': ['可适当添加既往治疗的中西药物', ],
        'aux_exam_prompts': ['可生成额外的支持诊断或鉴别诊断的辅助检查内容', ],
        'past_med_his_prompts': [
            '如有必要，可适当添加有利于诊断或鉴别诊断的既往史、个人史、婚育史、月经史'],
        'diagnosis_prompts': ['若未提供诊断信息，则不书写诊断结果部分',
                              ], }

}

input_templates_4 = {
    'head': '结构化信息如下：',
    'task': {
        'task_prompts': ['将下列结构化信息改写成用自然语言书写的中医医案', ]
    },
    'requirements': {
        'basic_requirement_prompts': [
            '对结构化信息进行同义词转换后，改写成自然语言',
            '生成内容只涉及病史信息，禁止生成关于中医病因病机、辨证分析、四诊分析、辨证结果、诊疗计划、处方、方剂组成、煎服法、注意事项的内容',
            '不改变原信息含义',
            '文字避免不必要的重复',
            '书写用语符合医学规范，逻辑清晰',
        ],
        'style_prompts': [
            {'prompt': '将结构化信息改写为中医医案，模仿<space_0>的风格及格式书写，但禁止使用中医理论相关词汇，不写风格模仿具体对象',
             'space': [[['贺普仁', '焦树德', '靳瑞', '李今庸', '李可', '李克绍', '刘渡舟', '刘弼臣',
                         '路志正', '吕炳奎', '任应秋', '裘沛然', '尚天裕', '施杞', '石学敏', '王永炎',
                         '王琦', '颜德馨', '杨甲三', '印会河', '岳美中', '张琪', '赵炳南', '赵绍琴',
                         '周仲瑛', '朱良春'], 1],
                       ]},
            '不必保留原结构化信息的格式'],
        'chief_complaint_prompts': ['确保主诉精简规范，主诉症状或体征合计不超过3个',
                                    '主诉不超过20个字',
                                    '若未给定主诉，则从临床表现中选择最急需解决的症状或体征作为主诉，主诉未必是前几个临床表现',
                                    '若临床表现涉及多系统疾病，选择单个系统的疾病作为主要诊断，主要诊断与主诉相对应',
                                    '虽然中医信息常涉及多个系统的症状体征，但是主诉书写时禁止对应多个不相关的系统或西医诊断，而应对应单个系统或单个西医诊断，其余信息无需在主诉中体现',
                                    ],
        'sym_phy_prompts': [
            '结构化信息中临床表现为乱序，需以主诉为核心调整书写顺序，主次分明，相关的临床表现相邻',
            '参考病史信息和对应处方添加指定的额外信息，但添加的额外信息不影响处方结果',
            [
                '适当添加与当前辨证论治无关的症状体征信息，但不影响最终辨证结果，不得额外增加支持当前辨证论治结果的其他症状体征',
                ''],
            '按病例书写习惯在刻下症处添加适当的阴性信息',
            '若未指定一般情况、进食、睡眠、大便、小便、查体，需按病例书写习惯补充相关阴性信息',
            '若个别临床表现与整体明显不符，可删去',
            '若舌象或脉象信息存在矛盾，需考虑是不同部位存在不同的舌象脉象的可能'],
        'pre_treat_prompts': ['可适当添加既往治疗的中西药物', ],
        'aux_exam_prompts': ['可生成额外的支持诊断或鉴别诊断的辅助检查内容', ],
        'past_med_his_prompts': ['若结构化信息中未指定既往史，需按病例书写习惯补充相关阴性信息'],
        'explanation_prompts': ['禁止添加关于本病例文本如何被编写的“元说明”和自我评价']
    }

}

input_templates_5 = {
    'head': '结构化信息如下：',
    'task': {
        'task_prompts': ['将下列结构化信息改写成用自然语言书写的中医医案', ]
    },
    'requirements': {
        'basic_requirement_prompts': [
            '对结构化信息进行同义词转换后，改写成自然语言',
            '生成内容只涉及病史信息，禁止生成关于中医病因、中医病机、辨证分析、四诊分析、辨证结果、诊疗计划、处方、方剂组成、煎服法、注意事项的内容',
            '不改变原信息含义',
            '文字避免不必要的重复，避免前后文自相矛盾',
            '书写用语符合医学规范，逻辑清晰',
        ],
        'style_prompts': [
            {
                'prompt': '将结构化信息改写为中医医案，模仿<space_0>的语言风格，但禁止使用中医理论相关词汇、严禁保留原结构化信息的格式，不需按照主诉、现病史、既往史等的顺序书写',
                'space': [[[
                            '陈修园', '程国彭', '傅青主', '高秉钧', '高士宗', '何梦瑶', '黄元御', '柯琴',
                            '雷丰', '李用粹', '林珮琴', '陆懋修', '沈金鳌', '唐宗海', '汪昂', '王清任',
                            '吴鞠通', '吴谦', '徐灵胎', '叶天士', '尤在泾', '喻嘉言', '张璐', '张志聪', '赵学敏',
                            '周学海', '曹炳章', '陈存仁', '陈无咎', '承淡安', '丁甘仁', '费伯雄', '何廉臣', '黄竹斋',
                            '恽铁樵', '陆渊雷', '秦伯未', '冉雪峰', '施今墨', '时逸人', '谭次仲', '唐容川',
                            '王孟英', '吴瑞甫', '谢观', '杨则民', '张山雷', '张锡纯', '章次公',
                            '吉益东洞', '丹波康赖', '曲直濑道三', '后藤艮山', '香川修庵', '山胁东洋',
                            '多纪元简', '多纪元胤', '浅田宗伯', '大塚敬节', '龙野一雄', '矢数道明', '森道伯',
                            '汤本求真',
                            '和田正系', '中神琴溪', '尾台榕堂', '永富独啸庵', '原南阳', '本间枣轩',
                            ], 1],
                          ]}, ],
        'sym_phy_prompts': [
            '若未给定主诉，则从临床表现中选择最急需解决的症状或体征作为主诉，主诉未必是前几个临床表现',
            '结构化信息中临床表现为乱序，需以主诉为核心调整书写顺序，主次分明，相关的临床表现相邻',
            '参考病史信息和对应处方添加指定的额外信息，但添加的额外信息不影响处方结果',
            [
                '适当添加与当前辨证论治无关的症状体征信息，但不影响最终辨证结果，不得额外增加支持当前辨证论治结果的其他症状体征',
                ''],
            '按病例书写习惯添加适当的阴性信息',
            '若个别临床表现与整体明显不符，可删去',
            '若舌象或脉象信息存在矛盾，需考虑是不同部位存在不同的舌象脉象的可能'],
        'pre_treat_prompts': ['可适当添加既往治疗', ],
        'explanation_prompts': ['禁止添加关于本病例文本如何被编写的“元说明”和自我评价']}

}

output_template = {
    'head': '医案及处方信息如下：',
    'task': {
        'task_prompts': ['根据给定的中医医案及处方信息，从理、法、方、药多个角度分析医案，并进行中医鉴别诊断，回顾相关中医知识，最后规范写出带有剂量的处方、煎服法及注意事项',]
    },
    'requirements': {
        'basic_requirement_prompts': [
            '不改变原信息含义',
            '用现代中医师的语言风格及格式书写',
        ],
        'diagnosis_prompts': ['除非提及中西医诊断，否则不书写任何诊断结果'],
        'mdf_exp': [
            '分析方药时，若结构化信息提供了加减法注释，将加减法注释改写精简且尽可能包含原信息，体现药证，可适当根据加减法注释推理加减法部分的中医病机'],
        'prescriptions': ['书写处方时不加减方药，不改变原方药、不改变方药剂量、不改变方药剂量单位、不改变煎服法'],
        'notes': ['可适当补充必要的注意事项'],
        'explanation_prompts': ['禁止添加关于本病例文本如何被编写的“元说明”和自我评价',
                                '生成内容中禁止提到任何上述文本生成要求的内容']
    }
}


class Root:
    def __init__(self, dir: str, root_path: str = None, contradictory_path: str = None):
        self.dir = dir
        self.root_path = os.path.join(self.dir, 'root.json') if root_path is None else root_path
        self.contradictory_path = os.path.join(self.dir,
                                               'contradictory.json') if contradictory_path is None else contradictory_path
    def get_rule_element_list(self):
        with open(self.root_path, 'r') as f:
            content = f.readline()
            content = json.loads(content)
        return content

    def get_contradictory(self):
        with open(self.contradictory_path, 'r') as f:
            content = f.readline()
            content = json.loads(content)
        return content


class PairedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.data[idx] = json.loads(self.data[idx])
        return self.data[idx][0], self.data[idx][1]


class Util:
    @staticmethod
    def _in_range(num, start, end):
        return start <= num < end

    @staticmethod
    def _make_dir(*paths):
        for path in paths:
            if os.path.exists(path):
                continue
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def _is_contradictory_inst(inst: Union[tuple, list, set], contradictory_elements_dict: dict):
        for elem in inst:
            if elem in contradictory_elements_dict.keys():
                contradictory_elems = contradictory_elements_dict[elem]
                if bool(set(inst) & set(contradictory_elems)):
                    return True
        return False

    @staticmethod
    def _save_cases(save_path: str, cases: list):
        with open(save_path, 'w') as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
            f.flush()

    @staticmethod
    def _save_to_one_dataset(save_path: str, gen_list: list, total_num: int, batch_size=30000000):
        if total_num <= batch_size:
            container = []
            for _, gen in gen_list:
                container.extend([case for case in gen])
            random.shuffle(container)
            Util._save_cases(save_path, container)
        else:
            temp_paths, read_files = Util._temp_save_to_disk(gen_list, save_path, total_num, batch_size)
            with open(save_path, 'w') as f:
                while len(read_files) > 0:
                    read_file = random.choice(read_files)
                    line = read_file.readline()
                    if line:
                        f.write(json.dumps(line, ensure_ascii=False) + '\n')
                        f.flush()
                    else:
                        read_files.remove(read_file)
                        read_file.close()
            for temp_path in temp_paths:
                os.remove(temp_path)

    @staticmethod
    def _temp_save_to_disk(gen_list, save_dir, total_num, batch_size, thread_idx=None) -> tuple:
        batch_num = math.ceil(total_num / batch_size)
        adjust_bs = math.ceil(total_num / batch_num)
        gen_list_split = []
        for _ in range(batch_num):
            gen_batch = []
            count = 0
            for i, (num, gen) in enumerate(gen_list):
                if count >= adjust_bs:
                    break
                if count + num <= adjust_bs:
                    gen_batch.append(gen)
                    gen_list.pop(i)
                    count += num
                else:
                    gen_1, gen_2 = tee(gen, 2)
                    gen_batch.append(islice(gen_1, 0, adjust_bs - count))
                    num = num - (adjust_bs - count)
                    if num > 0:
                        gen_list[i] = ((num), islice(gen_2, adjust_bs - count))
                    else:
                        gen_list.pop(i)
                    count += (adjust_bs - count)
            gen_list_split.append(gen_batch)
        temp_paths = []
        for i, gen_batch in enumerate(gen_list_split):
            container = []
            for gen in gen_batch:
                container.extend([case for case in gen])
            random.shuffle(container)
            if thread_idx is not None:
                temp_path = os.path.join(save_dir, f'temp{thread_idx}_{i}.txt')
            else:
                temp_path = os.path.join(save_dir, f'temp{i}.txt')
            Util._save_cases(temp_path, container)
            temp_paths.append(temp_path)
        read_files = [open(temp_path, 'r') for temp_path in temp_paths]
        return temp_paths, read_files

    @staticmethod
    def _get_psc_ratio_dict(root_elems_list, reduction_factor):
        psc_ratio_dict = dict()
        for rule in root_elems_list:
            if rule[0][:10] not in psc_ratio_dict.keys():
                psc_ratio_dict[rule[0][:10]] = 0
            psc_ratio_dict[rule[0][:10]] += 1
        ratios = Util.adjust_above_mean(list(psc_ratio_dict.values()), reduction_factor)
        for (k, v), n_v in zip(psc_ratio_dict.items(), ratios):
            psc_ratio_dict[k] = n_v / v
        return psc_ratio_dict

    @staticmethod
    def adjust_above_mean(data, reduction_factor):
        data = np.array(data, dtype=float)
        mean = np.mean(data)
        above_mean = data > mean
        excess = data[above_mean] - mean
        adjusted_excess = excess * reduction_factor
        adjusted_data = data.copy()
        adjusted_data[above_mean] = mean + adjusted_excess
        normalized_data = adjusted_data / np.sum(adjusted_data)
        normalized_data = normalized_data.tolist()
        return normalized_data
@dataclass
class PromptTemplateGen:
    task: dict = field(default_factory=dict)
    requirements: dict = field(default_factory=dict)
    head:str = field(default='信息如下：')
    def __post_init__(self):
        self.static_task, self.non_static_task = self.classify_static_prompts(
            list(self.task.values()))
        self.static_requirements, self.non_static_requirements = self.classify_static_prompts(
            list(self.requirements.values()))

    @classmethod
    def classify_static_prompts(cls, template):
        static_requirements = []
        non_static_requirements = []
        for prompt_class in template:
            for i, prompt in enumerate(prompt_class):
                if isinstance(prompt, str):
                    static_requirements.append(prompt)
                else:
                    non_static_requirements.append(prompt)
        return static_requirements, non_static_requirements

    def process_non_static_prompts(self, requirements):
        for i, prompt in enumerate(requirements):
            if isinstance(prompt, list):
                requirements[i] = random.choice(prompt)
        for i, prompt in enumerate(requirements):
            if isinstance(prompt, dict):
                requirements[i] = self.process_prompt_dict(prompt)
        requirements = list(filter(lambda x: x != '', requirements))
        return requirements

    @staticmethod
    def process_prompt_dict(prompt_dict: dict):
        space_content = []
        for choice in prompt_dict['space']:
            if isinstance(choice[1], int):
                sample_num = choice[1]
            elif isinstance(choice[1], tuple):
                sample_num = random.randint(*choice[1])
            else:
                raise Exception
            random.shuffle(choice[0])
            sampled_elem = '、'.join(choice[0][:sample_num])
            space_content.append(sampled_elem)
        for i, elem in enumerate(space_content):
            space_to_fill = f'<space_{i}>'
            prompt_dict['prompt'] = prompt_dict['prompt'].replace(space_to_fill, elem)
        return prompt_dict['prompt']

    def get_prompts(self, static_requirements, non_static_requirements):
        non_static_share_requirements = self.process_non_static_prompts(copy.deepcopy(non_static_requirements))
        requirements = static_requirements + non_static_share_requirements
        requirements = list(filter(lambda x: x != '', requirements))
        return requirements

    def __iter__(self):
        return self

    def __next__(self):
        task = self.get_prompts(self.static_task,
                                self.non_static_task)
        requirements = self.get_prompts(self.static_requirements,
                                        self.non_static_requirements)
        task = '；'.join(task) + '。'
        requirements = '要求：' + '；'.join(requirements) + '。'
        return task, requirements

@dataclass
class StructuredCase:
    root_elements: list = field(default_factory=list)
    contradictory: dict = field(default_factory=dict)
    template_gens: list = field(default_factory=list)
    psc_ratio_dict: dict = field(default_factory=dict)
    item_map:dict = field(default_factory=dict)
    input_seq:list = field(default_factory=list)
    output_seq:list = field(default_factory=list)
    total_num_to_generate: int = field(default=10000000)
    sample_try_times: int = field(default=20)
    op_max_len: int = field(default=6)
    len1_prob: float = field(default=0.2)
    len2_prob: float = field(default=0.3)
    len3_prob: float = field(default=0.3)
    inst_min_len: int = field(default=6)
    multiple_tp_ratio: float = field(default=0.1)
    tp_max_per_trait: int = field(default=2)
    name_prob: float = field(default=0.5)
    sex_prob: float = field(default=0.5)
    age_prob: float = field(default=0.5)
    time_of_onset_prob: float = field(default=0.5)
    aggravating_prob: float = field(default=0.5)
    max_aggravating: int = field(default=1)
    inducement_prob: float = field(default=0.5)
    max_inducement: int = field(default=1)
    pre_treat_prob: float = field(default=0.5)
    max_pre_treat: int = field(default=1)
    aux_exam_prob: float = field(default=0.0)
    max_aux_exam: int = field(default=1)
    disease_prob: float = field(default=0.5)
    max_disease: int = field(default=1)
    past_med_his_prob: float = field(default=0.0)
    max_past_med_his: int = field(default=1)
    common_prob: float = field(default=0)
    max_common: int = field(default=2)
    neg_prob: float = field(default=0.0)
    max_neg: int = field(default=2)
    add_note: bool = field(default=True)
    add_ref: bool = field(default=False)
    add_psc: bool = field(default=True)
    add_med: bool = field(default=True)
    add_mdf_exp: bool = field(default=True)

    def __post_init__(self):
        assert self.op_max_len >= 4 or self.len1_prob + self.len2_prob + self.len3_prob >= 1.0

        self.len1_interval = [0, self.len1_prob]
        self.len2_interval = [self.len1_prob, self.len1_prob + self.len2_prob]
        self.len3_interval = [self.len1_prob + self.len2_prob, self.len1_prob + self.len2_prob + self.len3_prob]

        psc_num, rule_num, chief_complaint, cpl, opt, neg, tong_pul, psc, med, note, ref, mdf_lookup_tab = self.root_elements
        self.psc_num = psc_num
        self.rule_num = rule_num
        self.psc = psc
        self.med = self._prepare_med(med)
        self.note = note
        self.ref = ref
        self.mdf_lookup_tab = mdf_lookup_tab
        self.chief_complaint = self._prepare_chief_complaint(chief_complaint)
        self.cpl = self._prepare_compulsory(cpl)
        self.opts, self.n_optionals_with_null = self._prepare_optionals(opt)
        self.tong_pul, self.tp_cpl_lib, self.tp_opt_lib, self.tp_depend = self._prepare_tongue_pulse(tong_pul)

        self.num_to_generate = self._get_num_to_generate()
        self.chief_complaint_gen = self._sample_chief_complaint()
        self.compulsory_gen = self._sample_compulsory()
        self.optional_gen = self._sample_optionals()
        self.tong_pul_gen = self._sample_tong_puls(self.multiple_tp_ratio, self.tp_max_per_trait)
        self.name_gen = self._sample_mingzi(self.name_prob)
        self.sex_gen = self._sample_special(opt[-10], prob=self.sex_prob)
        self.age_gen = self._sample_special(opt[-9], prob=self.age_prob, sample_mod='sample_from_range')
        self.time_of_onset_gen = self._sample_special(opt[-8], prob=self.time_of_onset_prob,
                                                      sample_mod='sample_from_range')
        self.aggravating_gen = self._sample_special(opt[-7], self.max_aggravating, self.aggravating_prob)
        self.inducement_gen = self._sample_special(opt[-6], self.max_inducement, self.inducement_prob)
        self.pre_treat_gen = self._sample_special(opt[-5], self.max_pre_treat, self.pre_treat_prob)
        self.aux_exam_gen = self._sample_special(opt[-4], self.max_aux_exam, self.aux_exam_prob)
        self.disease_gen = self._sample_special(opt[-3], self.max_disease, self.disease_prob)
        self.past_med_his_gen = self._sample_special(opt[-2], self.max_past_med_his, self.past_med_his_prob)
        self.common_gen = self._sample_special(opt[-1], self.max_common, self.common_prob)
        self.neg_gen = self._sample_special(neg, self.max_neg, self.neg_prob)

        self.root_elems_summery = self._get_root_elems_summery()

        if not self.item_map:
            self.item_map = {
                'name': '姓名', 'sex': '性别', 'age_sample': '年龄', 'chief_complaint': '主诉',
                'time_of_onset_sample': '发病持续时间',
                'inducement': '发病诱因', 'aggravating': '平时主诉加重原因', 'sym_phy': '临床表现', 'neg': '阴性症状',
                'tong_pul': '舌脉', 'pre_treat': '前期治疗', 'aux_exam': '辅助检查', 'disease': '疾病诊断',
                'past_med_his': '既往史', 'psc': '处方', 'med': '药物', 'note': '备注', 'mdf_exp': '加减法注释',
                'exp': '注释'}

        if not self.input_seq:
            self.input_seq = ['name', 'sex', 'age_sample', 'chief_complaint', 'time_of_onset_sample', 'inducement',
                              'sym_phy', 'tong_pul', 'pre_treat', 'aux_exam', 'disease', 'past_med_his', 'psc', 'med']

        if not self.output_seq:
            self.output_seq = ['psc', 'med', 'note', 'mdf_exp', 'exp']

    def _get_root_elems_summery(self):
        _, _, chief_complaint, cpl, opt, neg, tong_pul, _, _, _, _, _ = copy.deepcopy(self.root_elements)
        chief_complaint = self._prepare_chief_complaint(chief_complaint)
        cpl = self._prepare_compulsory(cpl)
        opt, _ = self._prepare_optionals(opt, prepare_root_elem_summery=True)
        cpl = [] if cpl is None else cpl
        opt = [] if opt is None else opt

        tps, _, _, _ = self._prepare_tongue_pulse(tong_pul)
        tps = {k:set(filter(lambda x: x != 'null', v)) for k, v in tps.items()}
        return chief_complaint, cpl, opt, tps


    def _get_num_to_generate(self):
        return math.ceil(self.total_num_to_generate * self.psc_ratio_dict[self.psc_num[:10]])

    def _sample_and_check(self, gen, existing=None, elems_check_contradictory: set = None):
        for _ in range(self.sample_try_times):
            content = set() if existing is None else copy.deepcopy(existing)
            content.update(next(gen))
            if elems_check_contradictory:
                elems_check_contradictory.update(content)
            else:
                elems_check_contradictory = content
            if not self._is_contradictory(content, elems_check_contradictory):
                return content, False
        return existing, True

    def _sample_chief_complaint(self):
        while True:
            content = set()
            content.add(random.choice(self.chief_complaint))
            content = chain.from_iterable(x.split('&') for x in content)
            content = set(random.choice(x.split('/')) for x in content)
            content = set(filter(lambda x: x != '' and x != "''", content))
            yield content

    def _sample_compulsory(self):
        while True:
            yield self.cpl

    def _sample_optionals(self):
        while True:
            content = set()
            for optional in self.opts:
                flt = random.random()
                if Util._in_range(flt, *self.len1_interval):
                    k = min(1, len(optional))
                elif Util._in_range(flt, *self.len2_interval):
                    k = min(2, len(optional))
                elif Util._in_range(flt, *self.len3_interval):
                    k = min(3, len(optional))
                else:
                    if self.op_max_len >= 4:
                        if 'null' in optional:
                            k = min(random.randint(4, max(self.op_max_len, self.n_optionals_with_null)), len(optional))
                        else:
                            k = min(random.randint(4, self.op_max_len), len(optional))
                    else:
                        raise Exception
                content.update(list(np.random.choice(optional, k, replace=False)))
            content = set(chain.from_iterable(x.split('&') for x in content))
            content = set(random.choice(x.split('/')) for x in content)
            content = set(filter(lambda x: x != '' and x != "''", content))
            yield content

    def _sample_special(self, pool: Union[list, dict], max_num: int = 1, prob: float = 1.0, maj_prob: float = 0.9,
                        sample_mod: str = 'default'):
        if all((bool(prob), bool(pool))):
            return self._sample_special_(pool, max_num, prob, maj_prob, sample_mod)
        else:
            return None

    def _sample_special_(self, pool: dict, max_num: int = 1, prob: float = 1.0, maj_prob: float = 0.9,
                         sample_mod: str = 'default'):
        assert max_num >= 1
        if sample_mod == 'default':
            if max_num == 1:
                assert len(pool.keys()) == 1
            while True:
                elems = set()
                temp_max_num = max_num
                for lib in pool.keys():
                    if re.search(r'<[cC]>', lib):
                        for vs in pool[lib].values():
                            elems.update(vs)
                    elif lib == '<own>':
                        if random.random() < prob:
                            if 'maj' in pool[lib] and random.random() < maj_prob:
                                k_maj = random.randint(1, min(max_num, len(pool[lib]['maj'])))
                                elems.update(list(np.random.choice(pool[lib]['maj'], k_maj, replace=False)))
                                temp_max_num -= k_maj
                            if temp_max_num > 0:
                                k_other = random.randint(1, min(max_num, len(pool[lib]['other'])))
                                elems.update(list(np.random.choice(pool[lib]['other'], k_other, replace=False)))
                    elif re.search(r'<[oO][a-zA-Z]*>', lib):
                        if 'maj' in pool[lib] and random.random() < maj_prob:
                            k_maj = random.randint(1, min(max_num, len(pool[lib]['maj'])))
                            elems.update(list(np.random.choice(pool[lib]['maj'], k_maj, replace=False)))
                            temp_max_num -= k_maj
                        if temp_max_num > 0:
                            k_other = random.randint(1, min(max_num, len(pool[lib]['other'])))
                            elems.update(list(np.random.choice(pool[lib]['other'], k_other, replace=False)))
                    else:
                        raise Exception
                yield elems if elems else None
        elif sample_mod == 'sample_from_range':
            # 从范围元素采样时，元素库里只有一组，而非多组
            assert len(pool.keys()) == 1 and max_num == 1
            k, v = next(iter(pool.items()))
            if k == '<own>':
                sample_with_pass = True
            elif re.search(r'<[cC]>', k) or re.search(r'<[oO][a-zA-Z]*>', k):
                sample_with_pass = False
            else:
                raise Exception
            while True:
                if not sample_with_pass:
                    if 'maj' in v and random.random() < maj_prob:
                        range_group = list(random.choice(v['maj']))
                    else:
                        range_group = list(random.choice(v['other']))
                else:
                    if random.random() < prob:
                        if 'maj' in v and random.random() < maj_prob:
                            range_group = list(random.choice(v['maj']))
                        else:
                            range_group = list(random.choice(v['other']))
                    else:
                        range_group = []
                elem, result = self._sample_from_time_range(range_group)
                yield elem, result
        else:
            raise Exception

    def _sample_from_time_range(self, range_group: list):
        if not range_group:
            return None, [[None, None]]
        tab = {'小时': (24, '天'), '天': (30, '月'), '月': (12, '年'), '年': (1, '岁'), '岁': (1, '年')}
        content = []
        annotation = ''
        for elem, mini, maxi, current_unit, maxi_unit in range_group:
            annotation += elem
            num = random.randint(mini, maxi)
            while current_unit != maxi_unit:
                temp = num / tab[current_unit][0]
                if temp >= 1:
                    num = temp
                    current_unit = tab[current_unit][1]
                else:
                    break
            if elem == '反复发作病程':
                content.append(['反复发作', str(round(num)) + current_unit])
            elif elem == '再发':
                content.append(['再发', str(round(num)) + current_unit])
            else:
                content.append(['', str(round(num)) + current_unit])
        return annotation, content

    def _sample_tong_puls(self, multiple_tp_ratio: float = 0.3, max_per_trait: int = 2):
        multiple_elem_dict = {'t_color': True, 't_nature': False, 't_coating_thickness': True,
                              't_coating_color': True, 't_coating_humidity': False, 't_coating_character': False,
                              'p_rate': False, 'p_rhythm': False, 'p_position': True, 'p_body': True,
                              'p_strength': True, 'p_fluency': True, 'p_tension': True, 'p_complex': False}
        assert max_per_trait >= 1
        multiple_elem_dict = {k: max_per_trait if v and random.random() < multiple_tp_ratio
        else 1 for k, v in multiple_elem_dict.items()}
        while True:
            remain_per_trait = multiple_elem_dict
            content = set()
            if self.tp_cpl_lib:
                content.update(chain.from_iterable(self.tp_cpl_lib.values()))
                for group_name in self.tp_cpl_lib.keys():
                    remain_per_trait[group_name] -= 1
            if self.tp_opt_lib:
                for tp_op in self.tp_opt_lib.values():
                    group_name = random.choice(list(tp_op.keys()))
                    content.update(np.random.choice(list(tp_op[group_name]), 1, replace=False))
                    remain_per_trait[group_name] -= 1
            for group_name in self.tong_pul.keys():
                if remain_per_trait[group_name] >= 1:
                    k = random.randint(1, min(remain_per_trait[group_name], len(self.tong_pul[group_name])))
                    content.update(np.random.choice(list(self.tong_pul[group_name]), k, replace=False))
            if self.tp_depend:
                independent_tps = content & set(self.tp_depend.keys())
                if independent_tps:
                    for independent_tp in independent_tps:
                        content.update([random.choice(xs.split('/')) for xs in self.tp_depend[independent_tp]])
            yield content

    @staticmethod
    def _sample_mingzi(prob):
        def _mask_name(name):
            surname = name[0]
            masked_part = '*' * (len(name) - 1)
            return surname + masked_part

        while True:
            if random.random() < prob:
                yield _mask_name(mingzi(female_rate=0.5)[0])
            else:
                yield ''

    def _process_med(self, med, case):
        if case['age']:
            if case['age'][0] == '新生儿期':
                med[1] = str(round(float(med[1]) * 0.17))
            elif case['age'][0] == '婴儿期':
                med[1] = str(round(float(med[1]) * 0.33))
            elif case['age'][0] == '幼儿期':
                med[1] = str(round(float(med[1]) * 0.50))
            elif case['age'][0] == '学龄前期':
                med[1] = str(round(float(med[1]) * 0.67))
            elif case['age'][0] == '学龄期':
                med[1] = str(round(float(med[1]) * 0.80))
            else:
                med[1] = str(round(float(med[1]) * 1.00))
        else:
            med[1] = str(round(float(med[1]) * 1.00))
        return med

    @classmethod
    def _tong_pul_to_dict(cls, t_color, t_nature, t_coating_thickness, t_coating_color, t_coating_humidity,
                          t_coating_character, p_rate, p_rhythm, p_position, p_body, p_strength, p_fluency,
                          p_tension, p_complex):
        result = {'t_color': t_color, 't_nature': t_nature, 't_coating_thickness': t_coating_thickness,
                  't_coating_color': t_coating_color, 't_coating_humidity': t_coating_humidity,
                  't_coating_character': t_coating_character, 'p_rate': p_rate, 'p_rhythm': p_rhythm,
                  'p_position': p_position, 'p_body': p_body, 'p_strength': p_strength,
                  'p_fluency': p_fluency, 'p_tension': p_tension, 'p_complex': p_complex}
        return result

    def _prepare_chief_complaint(self, chief_complaint):
        return chief_complaint if len(chief_complaint) >= 1 else None

    def _prepare_compulsory(self, compulsory):
        content = chain.from_iterable(filter(lambda x: len(x) >= 1, compulsory))
        content = chain.from_iterable(x.split('&') for x in content)
        content = set(random.choice(x.split('/')) for x in content)
        content = set(filter(lambda x: x != '' and x != "''", content))
        return content if len(content) >= 1 else None

    def _prepare_optionals(self, optionals, prepare_root_elem_summery:bool=False):
        temp = list(filter(lambda x: x is not None and len(x) > 0, optionals[:-10]))
        optionals_with_null = [elems_group for elems_group in temp if 'null' in elems_group]
        n_optionals_with_null = len(optionals_with_null)
        if not prepare_root_elem_summery:
            optionals_with_null = list(set(chain.from_iterable(optionals_with_null)))
            optional_list = list(filter(lambda x: 'null' not in x, temp))
            if optionals_with_null:
                optional_list.append(optionals_with_null)

            return (optional_list, n_optionals_with_null) if len(optional_list) >= 1 else (None, None)
        else:
            optional_list = [list(chain.from_iterable([x.split('&') for x in xs])) for xs in temp]
            optional_list = [['/'.join(filter(lambda z:z != "''", x.split('/'))) for x in xs] for xs in optional_list]
            optional_list = [list(filter(lambda x: x != 'null', xs)) for xs in optional_list]
            return (optional_list, n_optionals_with_null) if len(optional_list) >= 1 else (None, None)


    def _prepare_tongue_pulse(self, tongue_pulses):
        tong_pul = self._tong_pul_to_dict(*tongue_pulses)
        tps = dict()
        tp_cpl_lib = dict()
        tp_op_lib = dict()
        tp_depend = dict()
        for group_name in tong_pul.keys():
            tps[group_name] = set()
            for data in tong_pul[group_name]:
                data_split = data.replace('; ', '；').replace(';', '；').split('；')
                if 'F' in data_split[0]:
                    tps[group_name].add('null')
                else:
                    tps[group_name].add(data_split[0])

                if data_split[1]:
                    for cop in data_split[1].split('&'):
                        if re.search(r'[Cc]', cop):
                            if group_name not in tp_cpl_lib.keys():
                                tp_cpl_lib[group_name] = set()
                            tp_cpl_lib[group_name].add(data_split[0])
                        elif re.search(r'[Oo]', cop):
                            if cop not in tp_op_lib.keys():
                                tp_op_lib[cop] = dict()
                            if group_name not in tp_op_lib[cop].keys():
                                tp_op_lib[cop][group_name] = set()
                            tp_op_lib[cop][group_name].add(data_split[0])
                        else:
                            raise Exception
                depend_elems = data_split[2].split('&')
                if '' in depend_elems:
                    depend_elems.remove('')
                if depend_elems:
                    tp_depend[data_split[0]] = depend_elems
        if len(set(chain.from_iterable(tps.values()))) >= 1:
            return tps, tp_cpl_lib, tp_op_lib, tp_depend
        else:
            return (None,) * 4

    def _prepare_med(self, med):
        med = [[float(x) if i == 1 else x for i, x in enumerate(med_unit)] for med_unit in med]
        return med


    # ------------------过滤矛盾元素元素---------------------
    def _contradictory_elems_filter(self, current_elems, elems_to_add):
        content = set()
        for x in elems_to_add:
            if x in self.contradictory:
                if bool(set(current_elems) & set(self.contradictory[x])):
                    content.add(x)
        return list(set(elems_to_add) - content)

    def _is_contradictory(self, obj1, obj2=None):
        if obj2 is None:
            for x in obj1:
                if x in self.contradictory:
                    if bool(set(obj1) & set(self.contradictory[x])):
                        return True
            return False
        else:
            for x in obj2:
                if x in self.contradictory:
                    if bool(set(obj1) & set(self.contradictory[x])):
                        return True
            return False

    def _add_tokens(self, case: dict[str:Union[list, set]], tokens: Union[list, set, str], place: str,
                    filter_overlapping: bool = False, filter_contradictory: bool = False):
        if place not in case.keys():
            case[place] = list()
        if not tokens:
            return case
        if isinstance(tokens, str):
            tokens = [tokens]
        if filter_overlapping:
            tokens = list(filter(lambda x: x not in case, set(tokens)))
        if filter_contradictory:
            tokens = self._contradictory_elems_filter(tokens, tokens)
            tokens = self._contradictory_elems_filter(case[place], tokens)
        if isinstance(case[place], list):
            case[place].extend(tokens)
        elif isinstance(case[place], set):
            case[place].update(tokens)
        else:
            raise Exception
        return case

    def _get_case_generator(self):
        while self.num_to_generate:
            break_flag = False
            case = dict()
            sym_phy = set()
            if self.cpl is None and self.opts is None and self.tong_pul is None:
                logger.warn(f'方剂编号{self.psc_num}, 规则编号{self.rule_num}内容为空，予跳过')
                break
            if self.name_gen:
                name = next(self.name_gen)
                case = self._add_tokens(case, name, 'name')
            if self.sex_gen:
                sex = next(self.sex_gen)
                case = self._add_tokens(case, sex, 'sex')
            if self.age_gen:
                age_annotation, age_sample = next(self.age_gen)
                case = self._add_tokens(case, age_annotation, 'age')
                case = self._add_tokens(case, age_sample[0][1], 'age_sample')
            if self.time_of_onset_gen:
                time_of_onset_annotation, time_of_onset_sample = next(self.time_of_onset_gen)
                time_of_onset_sample = '，'.join([''.join(seq) for seq in time_of_onset_sample])
                case = self._add_tokens(case, time_of_onset_annotation, 'time_of_onset')
                case = self._add_tokens(case, time_of_onset_sample, 'time_of_onset_sample')
            if self.inducement_gen:
                inducement = next(self.inducement_gen)
                case = self._add_tokens(case, inducement, 'inducement')
            if self.aggravating_gen:
                aggravating = next(self.aggravating_gen)
                case = self._add_tokens(case, aggravating, 'aggravating')
            if self.chief_complaint:
                chief_complaint = next(self.chief_complaint_gen)
                sym_phy.update(chief_complaint)

                case = self._add_tokens(case, chief_complaint, 'chief_complaint')
            if self.cpl:
                sym_phy.update(next(self.compulsory_gen))
            if self.opts:
                places_check_contradictory = ['sex', 'age']
                elems_check_contradictory = set(
                    chain.from_iterable([case[place] for place in places_check_contradictory]))
                sym_phy, break_flag = self._sample_and_check(self.optional_gen, sym_phy, elems_check_contradictory)
            if sym_phy:
                case['sym_phy'] = sym_phy
            if self.tong_pul:
                tong_pul, break_flag = self._sample_and_check(self.tong_pul_gen)
                case['tong_pul'] = tong_pul
            if self.pre_treat_gen:
                pre_treat = next(self.pre_treat_gen)
                case = self._add_tokens(case, pre_treat, 'pre_treat')
            if break_flag:
                logger.warn(f'方剂编号{self.psc_num}, 规则编号{self.rule_num}规则矛盾，该规则生成跳过')
                break
            case = {k: list(filter(lambda x: x != 'null', v)) for k, v in case.items()}
            if self.aux_exam_gen:
                aux_exam = next(self.aux_exam_gen)
                case = self._add_tokens(case, aux_exam, 'aux_exam')
            if self.disease_gen:
                disease = next(self.disease_gen)
                case = self._add_tokens(case, disease, 'disease')

            if not any([case[place] for place in ['sym_phy', 'tong_pul', 'disease'] if place in case]):  # 若全为空，则跳过
                continue

            if self.neg_gen:
                neg = next(self.neg_gen)
                case = self._add_tokens(case, neg, 'sym_phy', True, False)
            if self.past_med_his_gen:
                past_med_his = next(self.past_med_his_gen)
                case = self._add_tokens(case, past_med_his, 'past_med_his', True, False)

            case = {k: list(set(filter(lambda x: x != 'null', v))) for k, v in case.items()}

            if self.add_psc and self.psc:
                case['psc'] = self.psc
            if self.add_med and self.med:
                med = [self._process_med(x, case) for x in copy.deepcopy(self.med)]
                case['med'] = [''.join(x) for x in med]
            if self.add_note and self.note:
                case['note'] = self.note
            if self.add_ref and self.ref:
                case['ref'] = self.ref
            if self.add_mdf_exp and self.mdf_lookup_tab:
                case['mdf_exp'] = self.add_mdf(case)
            case = {k : v for k, v in case.items() if v }
            case = {k: '，'.join(v) if k != 'mdf_exp' else v for k, v in case.items()}
            input_resource = [f"{self.item_map[item]}：{case[item]}" for item in self.input_seq if item in case.keys()]
            output_resource = [f"{self.item_map[item]}：{case[item]}" for item in self.output_seq if item in case.keys()]
            input_resource = '；'.join(input_resource) + '。'
            output_resource = '；'.join(output_resource) + '。'
            template_gen = random.choice(self.template_gens)
            task, requirements = next(template_gen)
            head = template_gen.head
            yield [task + head + input_resource + requirements, output_resource]
            self.num_to_generate -= 1

    def add_mdf(self, case):
        content = []
        mdf_lookup_tab = copy.deepcopy(self.mdf_lookup_tab)
        for mdf in mdf_lookup_tab:
            mdf_text = ''
            if mdf['增加元素目录']:
                overlap_elems = list(chain.from_iterable(
                    [set(case[column]) & set(mdf['增加元素']) for column in mdf['增加元素目录']]))
                if overlap_elems:
                    neg_elems = [x for x in overlap_elems if str(x).startswith('无')]
                    pos_elems = [x for x in overlap_elems if not str(x).startswith('无')]
                    if all([pos_elems, neg_elems]):
                        text = '因' + '、'.join(pos_elems) + '，且' + '、'.join(neg_elems) + '，'
                    else:
                        text = '因' + '、'.join(pos_elems) + '、'.join(neg_elems) + '，'
                    mdf_text += text
            if not mdf_text and mdf['删减元素']:
                temp = []
                temp.extend(random.sample(mdf['删减元素'], random.randint(1, min(2, len(mdf['删减元素'])))))
                if temp:
                    text = '因无' + '、'.join(temp) + '，'
                    mdf_text += text
            if mdf_text:
                temp = []
                if mdf['无剂量加减']:
                    temp.append('，'.join(mdf['无剂量加减']))
                if mdf['有剂量加减']:
                    mdf['有剂量加减'] = [''.join(self._process_med(x, case)) for x in mdf['有剂量加减']]
                    temp.append('用' + '，'.join(mdf['有剂量加减']))
                if temp:
                    mdf_text += '，'.join(temp)
                else:
                    raise Exception
                content.append(mdf_text)
        return '；'.join(content) if content else None

    def get_case_generator_wrapper(self):
        return self._case_generator_wrapper(), self.num_to_generate

    def _case_generator_wrapper(self):
        case_gen = self._get_case_generator()
        for case in case_gen:
            yield case

@dataclass
class CaseGenerator:
    root: Root
    input_template_gens: list = field(default_factory=list)
    output_template_gens: list = field(default_factory=list)
    generator_name: str = field(default='Gen')
    reduction_factor: float = field(default=0.1)
    max_room: int = field(default=500000000)
    shuffle: bool = field(default=True)
    std_cs_config: dict = field(default_factory=dict)
    model_name: str = field(default=None)
    llm_config: dict = field(default_factory=dict)
    sampling_params_config: dict = field(default_factory=dict)
    structured_case_path: str = field(default=None)
    simulated_case_path: str = field(default=None)
    processed_simulated_case_path: str = field(default=None)
    train_data_path:str = field(default=None)
    def __post_init__(self):
        self.root_elems_list = self.root.get_rule_element_list()
        self.contradictory = self.root.get_contradictory()
        self.psc_ratio_dict = Util._get_psc_ratio_dict(self.root_elems_list, self.reduction_factor)
        self.gen_dir = os.path.join(self.generator_name)
        self._temp_dir = os.path.join(self.generator_name, 'temp')
        if self.structured_case_path is None:
            self.structured_case_path = os.path.join(self.gen_dir, 'structured.jsonl')
        if self.simulated_case_path is None:
            self.simulated_case_path = os.path.join(self.gen_dir, 'simulated.jsonl')
        if self.processed_simulated_case_path is None:
            self.processed_simulated_case_path = os.path.join(self.gen_dir, 'processed_simulated.jsonl')
        if self.train_data_path is None:
            self.train_data_path = os.path.join(self.gen_dir, 'train_data.jsonl')

        Util._make_dir(self.gen_dir, self._temp_dir)
        if self.model_name:
            self.llm = LLM(
                model=self.model_name,
                **self.llm_config,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # 设置采样参数
            self.sampling_params = SamplingParams(
                **self.sampling_params_config
            )
        else:
            self.llm = None
        self.dataloader = None

    def generate_structured_cases(self, step: int = 1) -> None:
        gen_list = []
        total_num = 0
        for root_elements in self.root_elems_list:
            num, gen = self._make_structured_case_generator(root_elements, step)
            if num == 0:
                continue
            total_num += num
            gen_list.append((num, gen))
        Util._save_to_one_dataset(self.structured_case_path, gen_list, total_num, self.max_room)

    def _make_structured_case_generator(self, root_elements, step):
        inst = StructuredCase(root_elements, self.contradictory, self.input_template_gens, self.psc_ratio_dict,
                              **self.std_cs_config)
        case_generator, n_case = inst.get_case_generator_wrapper()
        if step > 1:
            n_case = math.floor(n_case / step)
            case_generator = islice(case_generator, 0, None, step)
        return n_case, case_generator

    def generate_with_LLM(self, mod, src_path=None, save_path=None):
        if mod == 'simulated_case':
            src_path = src_path if src_path is not None else self.structured_case_path
            save_path = save_path if save_path is not None else self.simulated_case_path
            self._generate_simulated_case_with_LLM(src_path, save_path)
        elif mod == 'output':
            src_path = src_path if src_path is not None else self.processed_simulated_case_path
            save_path = save_path if save_path is not None else self.train_data_path
            self._generate_output_with_LLM(src_path, save_path)
        else:
            raise Exception

    def _generate_simulated_case_with_LLM(self, src_path, save_path):
        with open(src_path, 'r') as f:
            dataset = PairedDataset(f.readlines())
        self.dataloader = DataLoader(dataset, batch_size=self.llm_config['max_num_seqs'], shuffle=self.shuffle)
        output_file = jsonlines.open(save_path, 'a')
        for data in tqdm(self.dataloader, desc='生成医案进度'):
            input_resources, output_resources = data
            input_resources = [[{"role": "user", "content": prompt}] for prompt in input_resources]
            input_resources = [self.tokenizer.apply_chat_template(prompt,
                                                          tokenize=False,
                                                          add_generation_prompt=True,
                                                          enable_thinking=False)
                       for prompt in input_resources]
            simulated_cases = self.llm.generate(input_resources, self.sampling_params)
            for simulated_case, output_resource in zip(simulated_cases, output_resources):
                output_file.write([simulated_case.outputs[0].text, output_resource])
        output_file.close()

    def _generate_output_with_LLM(self, src_path, save_path):
        with open(src_path, 'r') as f:
            dataset = PairedDataset(f.readlines())
        self.dataloader = DataLoader(dataset, batch_size=self.llm_config['max_num_seqs'], shuffle=self.shuffle)
        output_file = jsonlines.open(save_path, 'a')
        for data in tqdm(self.dataloader, desc='生成医案进度'):
            simulated_cases, output_resources = data
            output_resources = [[{"role": "user", "content": prompt}] for prompt in output_resources]
            output_resources = [self.tokenizer.apply_chat_template(prompt,
                                                          tokenize=False,
                                                          add_generation_prompt=True,
                                                          enable_thinking=False)
                       for prompt in output_resources]
            outputs = self.llm.generate(output_resources, self.sampling_params)
            for simulated_case, output in zip(simulated_cases, outputs):
                output_file.write({"input": simulated_case, 'output': output.outputs[0].text})
        output_file.close()

    def _process_output_resources(self, file_path):
        try:
            content = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    output_template_gen = random.choice(self.output_template_gens)
                    head = output_template_gen.head
                    task, requirements = next(output_template_gen)
                    data[1] = task + head + data[0] + data[1] + requirements
                    content.append([data[0], data[1]])

            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             delete=False, suffix='.json') as temp_f:
                json.dump(content, temp_f, ensure_ascii=False, indent=2)
                temp_path = temp_f.name
            backup_path = file_path + '.backup'
            os.rename(file_path, backup_path)

            try:
                os.rename(temp_path, file_path)
                print(f"✅ 文件处理完成: {file_path}")
                print(f"📁 备份文件: {backup_path}")
                return True
            except:
                os.rename(backup_path, file_path)
                raise

        except Exception as e:
            print(f"❌ 错误: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            return False


if __name__ == '__main__':
    model_name = 'Qwen/Qwen3-30B-A3B-Instruct-2507'

    llm_config = {'max_num_seqs': 80,
                  'max_num_batched_tokens': 240000,
                  'max_model_len': 3000,
                  'block_size': 16,  # 新增：KV缓存块大小
                  'tensor_parallel_size': 1,
                  'gpu_memory_utilization': 0.95,
                  'enforce_eager': False}

    sampling_params_config = {
        'temperature': 0.2,
        'top_p': 0.9,
        'top_k': 50,
        'max_tokens': 3000,
        'repetition_penalty': 1.1,
    }

    std_cs_config = {
        'total_num_to_generate': 200000,
        'sample_try_times': 20,
        'op_max_len': 6,
        'len1_prob': 0.2,
        'len2_prob': 0.3,
        'len3_prob': 0.3,
        'inst_min_len': 6,
        'multiple_tp_ratio': 0.1,
        'tp_max_per_trait': 2,
        'sex_prob': 0.5,
        'age_prob': 0.5,
        'time_of_onset_prob': 0.5,
        'aggravating_prob': 0.5,
        'max_aggravating': 1,
        'inducement_prob': 0.5,
        'max_inducement': 1,
        'pre_treat_prob': 0.5,
        'max_pre_treat': 1,
        'aux_exam_prob': 0.0,
        'max_aux_exam': 1,
        'disease_prob': 0.5,
        'max_disease': 1,
        'past_med_his_prob': 0.0,
        'max_past_med_his': 1,
        'common_prob': 0,
        'max_common': 2,
        'neg_prob': 0.7,
        'max_neg': 2,
        'add_note': False,
        'add_ref': False,
        'add_psc': True,
        'add_med': True,
        'add_mdf_exp': False,
    }
    root = Root(os.path.join('root'), root_path=os.path.join('root', 'root_mod.json'))

    iptgs = [PromptTemplateGen(**input_template_1),
             PromptTemplateGen(**input_templates_2),
             PromptTemplateGen(**input_templates_3),
             PromptTemplateGen(**input_templates_4),
             PromptTemplateGen(**input_templates_5),]


    optgs = [PromptTemplateGen(**output_template), ]
    # cg = CaseGenerator(root,
    #                    iptgs,
    #                    optgs,
    #                    shuffle=True,
    #                    std_cs_config=std_cs_config,
    #                    model_name=model_name,
    #                    llm_config=llm_config,
    #                    sampling_params_config=sampling_params_config,
    #                    )

    # cg = CaseGenerator(root, ptgs)
    # cg.generate_structured_cases()
    # cg.generate_with_LLM(mod='simulated_case')
    # cg._process_output_resources(os.path.join('Gen', 'simulated_processed'))
    # cg.generate_with_LLM(mod='output')
