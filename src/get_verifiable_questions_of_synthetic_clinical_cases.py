
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_fixed, stop_after_attempt
from openai import OpenAI
import math
import random
import copy
from dataclasses import dataclass, field
import numpy as np
import json
from mingzi import mingzi
from logger import get_logger
import pandas as pd
from itertools import chain
import os
import re
from typing import Union

os.environ['OMP_NUM_THREADS'] = '1'
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
torch.cuda.set_device(0)
device = "cuda"
logger = get_logger('cgenerate')

inp_prompt_template_1 = {
    'head': 'Structured information is provided as follows:',
    'task': {
        'task_prompts': [
            'Rewrite the following structured information into a TCM medical case in **Chinese**',
        ]
    },
    'requirements': {
        'basic_requirement_prompts': [
            'Paraphrase the structured information using synonyms and rewrite it in natural language',
            'The generated content shall only include medical history information; it is prohibited to generate content related to TCM etiology and pathogenesis, syndrome differentiation analysis, four-diagnosis analysis, syndrome differentiation results, diagnosis and treatment plans, prescriptions, formula compositions, administration methods, and precautions',
            'Do not alter the original meaning of the information',
            'Avoid unnecessary text repetition',
            'Use medically standardized terminology and ensure clear logic',
            'The generated content shall be logically consistent and realistic at each stage, with no contradictions in the context',
        ],
        'style_prompts': [
            'The content shall be written in the standard sequential format of Chief Complaint, History of Present Illness, Current Condition, Physical Examination, Past Medical History, Personal History, and Auxiliary Examinations',
            'For female patients, menstrual history and gynecological and obstetric history shall be additionally included',
            'The format of the original structured information need not be retained'],
        'chief_complaint_prompts': [
            'Ensure the chief complaint is concise and standardized, with no more than 3 symptoms or signs',
            'The chief complaint shall not exceed 20 Chinese characters',
            'If no chief complaint is provided, select the most urgent symptom or sign from the clinical manifestations as the chief complaint; the chief complaint is not necessarily among the first few clinical manifestations',
            'If the clinical manifestations involve multi-system diseases, select a single-system disease as the primary diagnosis, which shall correspond to the chief complaint',
            'Although TCM information often involves symptoms and signs of multiple systems, the chief complaint shall not correspond to multiple unrelated systems or Western medical diagnoses, but only to a single system or a single Western medical diagnosis; other information need not be reflected in the chief complaint',
        ],
        'sym_phy_prompts': [
            'The clinical manifestations in the structured information are in random order; adjust the writing sequence around the chief complaint, with clear priorities and adjacent arrangement of related clinical manifestations',
            'Add the specified additional information with reference to the medical history and corresponding prescriptions, provided that such additions do not affect the prescription outcomes; it is strictly prohibited to add additional information that may violate prescription contraindications',
            'Add appropriate negative findings to the Current Symptoms section in accordance with medical case-writing conventions',
            'If the general condition, appetite, sleep, bowel movements, urination, or physical examination are not specified, supplement the relevant negative information in accordance with medical case-writing conventions',
            'Individual clinical manifestations that are obviously inconsistent with the overall context may be deleted',
            'If there are contradictions in tongue or pulse manifestations, consider the possibility of different tongue/pulse manifestations in different parts'],
        'pre_treat_prompts': [
            'Chinese and Western medications used in previous treatments may be appropriately added in accordance with medical case-writing conventions', ],
        'aux_exam_prompts': [
            'Additional auxiliary examination content supporting diagnosis or differential diagnosis may be generated in accordance with medical case-writing conventions', ],
        'past_med_his_prompts': [
            'If the past medical history, personal history, gynecological and obstetric history, or menstrual history are not specified in the structured information, supplement the relevant negative information in accordance with medical case-writing conventions',
            'If necessary, appropriately add past medical history, personal history, gynecological and obstetric history, and menstrual history that are conducive to diagnosis or differential diagnosis in accordance with medical case-writing conventions',
            [{
                'prompt': 'The patient has a past history of underlying diseases related to <space_0>. The patient has <space_1>. Randomly add a past medical history for the patient',
                'space': [
                    [['Circulatory System', 'Respiratory System', 'Digestive System', 'Nervous System', 'Motor System',
                      'Urogenital System'], (1, 2)],
                    [['no previous surgery', 'a previous surgery', '2 previous surgeries'], 1],
                ]}, '', '', '', ]],
        'diagnosis_prompts': [
            'If no diagnostic information is provided, the diagnosis result section shall not be included',
        ],
        'explanation_prompts': [
            'It is prohibited to add "meta-explanations" or self-evaluations regarding how this case text is compiled',
            'Do not mention any of the above text generation requirements in the generated content']
    }
}

inp_prompt_template_2 = {
    'head': 'Structured information is provided as follows:',
    'task': {
        'task_prompts': [
            'Rewrite the following structured information into a TCM medical case in **Chinese**',
        ]
    },
    'requirements': {
        'basic_requirement_prompts': [
            'Paraphrase the structured information using synonyms and rewrite it in natural language',
            'The generated content shall only include medical history information; it is prohibited to generate content related to TCM etiology and pathogenesis, syndrome differentiation analysis, four-diagnosis analysis, syndrome differentiation results, diagnosis and treatment plans, prescriptions, formula compositions, administration methods, and precautions',
            'Do not alter the original meaning of the information',
            'Avoid unnecessary text repetition',
            'Use medically standardized terminology and ensure clear logic',
            'The generated content shall be logically consistent and realistic at each stage, with no contradictions in the context',
        ],
        'style_prompts': [
            'Imitate the format and linguistic style of TCM medical cases written by modern and contemporary TCM physicians; the format of the original structured information need not be retained, and there is no requirement to write in the order of Chief Complaint, History of Present Illness, Past Medical History, etc.'],
        'chief_complaint_prompts': [
            'Ensure the chief complaint is concise and standardized, with a total of no more than 3 symptoms or signs',
            'The chief complaint shall not exceed 20 Chinese characters',
            'If no chief complaint is provided, select the most urgent symptom or sign from the clinical manifestations as the chief complaint; the chief complaint is not necessarily among the first few clinical manifestations',
            'If the clinical manifestations involve multi-system diseases, select a single-system disease as the primary diagnosis, which shall correspond to the chief complaint',
            'Although TCM information often involves symptoms and signs of multiple systems, the chief complaint shall not correspond to multiple unrelated systems or Western medical diagnoses, but only to a single system or a single Western medical diagnosis; other information need not be reflected in the chief complaint',
        ],
        'sym_phy_prompts': [
            'The clinical manifestations in the structured information are in random order; adjust the writing sequence around the chief complaint, with clear priorities and adjacent arrangement of related clinical manifestations',
            'Add the specified additional information with reference to the medical history and corresponding prescriptions, provided that such additions do not affect the prescription outcomes; it is strictly prohibited to add additional information that may violate prescription contraindications',
            'Add appropriate negative findings to the Current Symptoms section in accordance with medical case-writing conventions',
            'If the general condition, appetite, sleep, bowel movements, urination, or physical examination are not specified, supplement the relevant negative information in accordance with medical case-writing conventions',
            'Individual clinical manifestations that are obviously inconsistent with the overall context may be deleted',
            'If there are contradictions in tongue or pulse manifestations, consider the possibility of different tongue/pulse manifestations in different parts'],
        'pre_treat_prompts': [
            'Chinese and Western medications used in previous treatments may be appropriately added', ],
        'aux_exam_prompts': [
            'Additional auxiliary examination content supporting diagnosis or differential diagnosis may be generated', ],
        'past_med_his_prompts': [
            'If necessary, appropriately add past medical history, personal history, gynecological and obstetric history, and menstrual history that are conducive to diagnosis or differential diagnosis',
            [{
                'prompt': 'The patient has a past history of underlying diseases related to <space_0>. The patient has <space_1>. Randomly add an irrelevant past medical history for the patient',
                'space': [
                    [['Circulatory System', 'Respiratory System', 'Digestive System', 'Nervous System', 'Motor System',
                      'Urogenital System'], (1, 2)],
                    [['no previous surgery', 'a previous surgery', '2 previous surgeries'], 1],
                ]}, '', '', '', ]],
        'diagnosis_prompts': [
            'If no diagnostic information is provided, the diagnosis result section shall not be included', ],
    }
}

inp_prompt_template_3 = {
    'head': 'Structured information is provided as follows:',
    'task': {
        'task_prompts': [
            'Rewrite the following structured information into a TCM medical case in **Chinese**',
        ]
    },
    'requirements': {
        'basic_requirement_prompts': [
            'Paraphrase the structured information using synonyms and rewrite it in natural language',
            'The generated content shall only include medical history information; it is prohibited to generate content related to TCM etiology and pathogenesis analysis, syndrome differentiation analysis, four-diagnosis analysis, syndrome differentiation results, diagnosis and treatment plans, prescriptions, formula compositions, administration methods, and precautions',
            'Do not alter the original meaning of the information',
            'Avoid unnecessary text repetition',
            'The generated content shall be logically consistent and realistic at each stage, with no contradictions in the context',
        ],
        'style_prompts': [
            'Convert the structured information into a colloquial TCM medical case',
            'The format of the original structured information need not be retained',
        ],
        'chief_complaint_prompts': [
            'If no chief complaint is provided, select the most urgent symptom or sign from the clinical manifestations as the chief complaint; the chief complaint is not necessarily among the first few clinical manifestations',
            'If the clinical manifestations involve multi-system diseases, select a single-system disease as the primary diagnosis, which shall correspond to the chief complaint',
            'Although TCM information often involves symptoms and signs of multiple systems, the chief complaint shall not correspond to multiple unrelated systems or Western medical diagnoses, but only to a single system or a single Western medical diagnosis; other information need not be reflected in the chief complaint',
        ],
        'sym_phy_prompts': [
            'The clinical manifestations in the structured information are in random order; adjust the writing sequence around the chief complaint, with clear priorities and adjacent arrangement of related clinical manifestations',
            'Add the specified additional information with reference to the medical history and corresponding prescriptions, provided that such additions do not affect the prescription outcomes; it is strictly prohibited to add additional information that may violate prescription contraindications',
            'Add appropriate negative findings to the Current Symptoms section in accordance with medical case-writing conventions',
            'If the general condition, appetite, sleep, bowel movements, urination, or physical examination are not specified, supplement the relevant negative information in accordance with medical case-writing conventions',
            'Individual clinical manifestations that are obviously inconsistent with the overall context may be deleted',
            'If there are contradictions in tongue or pulse manifestations, consider the possibility of different tongue/pulse manifestations in different parts'],
        'pre_treat_prompts': [
            'Chinese and Western medications used in previous treatments may be appropriately added', ],
        'aux_exam_prompts': [
            'Additional auxiliary examination content supporting diagnosis or differential diagnosis may be generated', ],
        'past_med_his_prompts': [
            'If necessary, appropriately add past medical history, personal history, gynecological and obstetric history, and menstrual history that are conducive to diagnosis or differential diagnosis'],
        'diagnosis_prompts': [
            'If no diagnostic information is provided, the diagnosis result section shall not be included',
        ], }

}

inp_prompt_template_4 = {
    'head': 'Structured information is provided as follows:',
    'task': {
        'task_prompts': [
            'Rewrite the following structured information into a TCM medical case in **Chinese**',
        ]
    },
    'requirements': {
        'basic_requirement_prompts': [
            'Paraphrase the structured information using synonyms and rewrite it in natural language',
            'The generated content shall only include medical history information; it is prohibited to generate content related to TCM etiology and pathogenesis, syndrome differentiation analysis, four-diagnosis analysis, syndrome differentiation results, diagnosis and treatment plans, prescriptions, formula compositions, administration methods, and precautions',
            'Do not alter the original meaning of the information',
            'Avoid unnecessary text repetition',
            'Use medically standardized terminology',
            'The generated content shall be logically consistent and realistic at each stage, with no contradictions in the context',
        ],
        'style_prompts': [
            {
                'prompt': 'Imitate the style and format of <space_0>, but prohibit the use of vocabulary related to TCM theory and do not specify the subject being imitated',
                'space': [[['贺普仁', '焦树德', '靳瑞', '李今庸', '李可', '李克绍', '刘渡舟', '刘弼臣',
                            '路志正', '吕炳奎', '任应秋', '裘沛然', '尚天裕', '施杞', '石学敏', '王永炎',
                            '王琦', '颜德馨', '杨甲三', '印会河', '岳美中', '张琪', '赵炳南', '赵绍琴',
                            '周仲瑛', '朱良春'], 1],
                          ]},
            'The format of the original structured information need not be retained; and there is no requirement to write in the order of Chief Complaint, History of Present Illness, Past Medical History, etc.'
        ],
        'chief_complaint_prompts': [
            'Ensure the chief complaint is concise and standardized, with a total of no more than 3 symptoms or signs',
            'The chief complaint shall not exceed 20 Chinese characters',
            'If no chief complaint is provided, select the most urgent symptom or sign from the clinical manifestations as the chief complaint; the chief complaint is not necessarily among the first few clinical manifestations',
            'If the clinical manifestations involve multi-system diseases, select a single-system disease as the primary diagnosis, which shall correspond to the chief complaint',
            'Although TCM information often involves symptoms and signs of multiple systems, the chief complaint shall not correspond to multiple unrelated systems or Western medical diagnoses, but only to a single system or a single Western medical diagnosis; other information need not be reflected in the chief complaint',
        ],
        'sym_phy_prompts': [
            'The clinical manifestations in the structured information are in random order; adjust the writing sequence around the chief complaint, with clear priorities and adjacent arrangement of related clinical manifestations',
            'Add the specified additional information with reference to the medical history and corresponding prescriptions, provided that such additions do not affect the prescription outcomes; it is strictly prohibited to add additional information that may violate prescription contraindications',
            'Add appropriate negative findings to the Current Symptoms section in accordance with medical case-writing conventions',
            'If the general condition, appetite, sleep, bowel movements, urination, or physical examination are not specified, supplement the relevant negative information in accordance with medical case-writing conventions',
            'Individual clinical manifestations that are obviously inconsistent with the overall context may be deleted',
            'If there are contradictions in tongue or pulse manifestations, consider the possibility of different tongue/pulse manifestations in different parts'],
        'pre_treat_prompts': [
            'Chinese and Western medications used in previous treatments may be appropriately added', ],
        'aux_exam_prompts': [
            'Additional auxiliary examination content supporting diagnosis or differential diagnosis may be generated', ],
        'past_med_his_prompts': [
            'If the past medical history is not specified in the structured information, supplement the relevant negative information in accordance with medical case-writing conventions'],
        'explanation_prompts': [
            'It is prohibited to add "meta-explanations" or self-evaluations regarding how this case text is compiled']
    }
}

verify_prompt = """<data>
{}
</data>
Judge and output "false" if any of the following situations exist in the data: 
1. Obvious common sense errors;
2. Internal logical contradictions;
3. Prescriptions that violate TCM compatibility contraindications, usage taboos and clinical medication norms; 
If none of the above situations exist, output "true".
Output only "true" or "false" without any extra explanation.
"""

clean_simulated_case_prompt = """<MedicalCase>
{}
</MedicalCase>

You are given a TCM clinical case (<MedicalCase>). Your task is to clean and reorganize it into a **TCM test question** that:
- Contains NO answers, NO hints to answers, NO diagnostic conclusions.
- Contains NO irrelevant information that does not belong to a standard TCM medical case.

You must remove at minimum:
1. Content related to etiology, pathogenesis, syndrome differentiation reasoning, syndrome differentiation results, diagnosis, treatment principles, prescriptions, or any therapeutic plan.
2. Any extraneous, non-medical, or irrelevant text.

The original case may be written in formal medical style, colloquial style, or classical Chinese; this is acceptable and does not need correction.

Output only the cleaned medical case text, without any extra explanation.
"""

class LLMClient:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def call_llm(self, prompt_content, generation_params=None):
        if generation_params is None:
            generation_params = {
                'temperature': 0.4,
                'top_p': 0.9,
                'max_tokens': 3000,
                'repetition_penalty': 1.05,
            }

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_content}],
            # **generation_params,
            timeout=300
        )
        response_content = chat_completion.choices[0].message.content
        return response_content

    @retry(wait=wait_fixed(5000), stop=stop_after_attempt(5), reraise=True)
    def retry_call(self, prompt_content, generation_params=None):
        return self.call_llm(prompt_content, generation_params)

@dataclass
class PromptTemplateGen:
    task: dict = field(default_factory=dict)
    requirements: dict = field(default_factory=dict)
    head: str = field(default='The information is as follows:')

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
        task = '. '.join(task) + '.'
        requirements = 'Requirements: ' + '. '.join(requirements) + '.'
        return task, requirements

@dataclass
class StructuredCase:
    root_elements: dict = field(default_factory=list)
    psc_ratio_dict: dict = field(default_factory=dict)
    total_num_to_generate: int = field(default=100)
    item_map: dict = field(default_factory=dict)
    input_seq: list = field(default_factory=list)
    output_seq: list = field(default_factory=list)
    sample_num_probs: dict = field(default_factory=lambda:{1:0.20, 2:0.55, 3:0.90, 4:0.95, 5:1.0})
    skip_sample_coef:float = field(default=1.0)
    sample_maj_prob:float = field(default=0.9)
    max_num_dict:dict = field(default_factory=dict)
    add_note: bool = field(default=False)
    add_ref: bool = field(default=False)
    add_psc: bool = field(default=True)
    add_med: bool = field(default=True)

    def __post_init__(self):
        assert list(self.sample_num_probs.values()) == sorted(self.sample_num_probs.values())
        self.psc_num = self.root_elements['prescription_num']
        self.rule_num = self.root_elements['rule_num']
        self.psc = self.root_elements['prescription']
        self.med = self._prepare_med(self.root_elements['medicines'])
        self.note = self.root_elements['note']
        self.ref = self.root_elements['reference']

        self.chief_complaint_gen = self.get_sampler(self.root_elements['classification'])
        self.sym_signs_gen = self.get_sampler(self.root_elements['sym_signs'])
        self.tong_pul_gen = self.get_sampler(self.root_elements['tongue_pulse'])
        self.name_gen = self._get_name_sampler()
        self.sex_gen = self.get_sampler(self.root_elements['sex'])
        self.age_gen = self.get_sampler_for_range_elems(self.root_elements['age'])
        self.time_of_onset_gen = self.get_sampler_for_range_elems(self.root_elements['time_of_onset'])
        self.aggravating_gen = self.get_sampler(self.root_elements['aggravating'])
        self.inducement_gen = self.get_sampler(self.root_elements['inducement'])
        self.pre_treat_gen = self.get_sampler(self.root_elements['previous_treatment'])
        self.aux_exam_gen = self.get_sampler(self.root_elements['auxiliary_examination'])
        self.disease_gen = self.get_sampler(self.root_elements['diagnosis'])
        self.past_med_his_gen = self.get_sampler(self.root_elements['past_med_history'])
        self.cross_libs_gen = self.get_sampler(self.root_elements['cross_libs'])
        self.cross_lib_elems = set()
        self.num_to_generate = self._get_num_to_generate()

        if not self.item_map:
            self.item_map = {
                'name': '姓名', 'sex': '性别', 'age_sample': '年龄', 'chief_complaint': '主诉',
                'time_of_onset_sample': '发病持续时间',
                'inducement': '发病诱因', 'aggravating': '平时主诉加重原因', 'sym_phy': '临床表现', 'neg': '阴性症状',
                'tong_pul': '舌脉', 'pre_treat': '前期治疗', 'aux_exam': '辅助检查', 'disease': '疾病诊断',
                'past_med_his': '既往史', 'psc': '处方', 'med': '药物', 'note': '备注', 'exp': '注释'}

        if not self.input_seq:
            self.input_seq = ['name', 'sex', 'age_sample', 'chief_complaint', 'time_of_onset_sample', 'inducement',
                              'sym_phy', 'tong_pul', 'pre_treat', 'aux_exam', 'disease', 'past_med_his', 'psc', 'med']

        if not self.output_seq:
            self.output_seq = ['psc', 'med', ]

    @staticmethod
    def _in_range(num, start, end):
        return start <= num < end

    def _get_num_to_generate(self):
        return math.ceil(self.total_num_to_generate * self.psc_ratio_dict[self.psc_num[:10]])

    def _sample_elems(self, lib, sample_max_num:int=None, sample_maj_prob:float=0.9):
        elems = set()
        flt = random.random()
        k = 0
        lib_elems_num = len(list(chain.from_iterable(lib.values())))
        if sample_max_num is None:
            sample_max_num = lib_elems_num
        for sample_num, prob in self.sample_num_probs.items():
            if flt < prob:
                k = min(sample_num, sample_max_num, lib_elems_num)
                break
        cross_lib_intersection = set(chain.from_iterable([v & self.cross_lib_elems for k, v in lib.items()]))
        k -= len(cross_lib_intersection)
        if k < 0:
            elem_to_remove = set(np.random.choice(list(cross_lib_intersection), abs(k)))
            cross_lib_intersection -= elem_to_remove
            self.cross_lib_elems -= elem_to_remove
        else:
            remain_lib = {k: v - cross_lib_intersection for k, v in lib.items()}
            for _ in range(k):
                if 'other' in lib and random.random() > sample_maj_prob:
                    sampled_elem = random.choice(list(remain_lib['other']))
                else:
                    sampled_elem = random.choice(list(remain_lib['maj']))
                elems.add(sampled_elem)
        if cross_lib_intersection:
            elems.update(cross_lib_intersection)
        return elems

    def _get_skip_sample_prob(self, lib_len:int):
        return 1 - np.clip(0.1 * lib_len * self.skip_sample_coef, 0.05, 0.95)

    def get_sampler(self, libs: dict, max_sample_num: int = None):
        if not libs:
            return None
        return self._get_sampler(libs, max_sample_num)

    def _get_sampler(self, libs: dict, max_sample_num: int = None):
        if max_sample_num is not None:
            assert max_sample_num >= 1
        while True:
            elems = set()
            for lib in libs.keys():
                if re.search(r'(c?[Cc])\d+', lib):
                    for vs in libs[lib].values():
                        elems.update(vs)
                elif re.search(r'(c?[Oo]z?)\d+', lib):
                    skip_sample_prob = self._get_skip_sample_prob(len(lib))
                    if re.search(r'(c?[Oo]z)\d+', lib) and random.random() < skip_sample_prob:
                        continue
                    sampled_elems = self._sample_elems(libs[lib], None, self.sample_maj_prob)
                    elems.update(sampled_elems)
                elif re.search(r'(c?[Aa]z?)\d+', lib):
                    skip_sample_prob = self._get_skip_sample_prob(len(lib))
                    if re.search(r'(c?[Aa]z)\d+', lib) and random.random() < skip_sample_prob:
                        continue
                    sampled_elems = self._sample_elems(libs[lib], 1, self.sample_maj_prob)
                    elems.update(sampled_elems)
                else:
                    raise Exception
            yield elems if elems else None

    def get_sampler_for_range_elems(self, libs: dict, max_sample_num: int = 1):
        assert len(libs.keys()) == 1 and max_sample_num == 1
        k, v = next(iter(libs.items()))
        while True:
            skip_sample_prob = self._get_skip_sample_prob(5)
            if re.search(r'([OoAa]z)\d+', k) and random.random() < skip_sample_prob:
                trait, range_group = '', []
            elif re.search(r'([OoAaCc]|[OoAa]z)\d+', k):
                if 'other' in v and random.random() > self.sample_maj_prob:
                    trait, range_group = list(random.choice(v['other']))
                else:
                    trait, range_group = list(random.choice(v['maj']))
            else:
                trait, range_group = '', []
            result = self._sample_from_time_range(range_group)
            yield trait, result

    def _sample_from_time_range(self, range_group: list):
        if not range_group:
            return [['', '']]
        tab = {'小时': (24, '天'), '天': (30, '月'), '月': (12, '年'), '年': (1, '岁'), '岁': (1, '年')}
        content = []
        for elem, mini, maxi, current_unit, maxi_unit in range_group:
            num = random.randint(mini, maxi)
            while current_unit != maxi_unit:
                temp = num / tab[current_unit][0]
                if temp >= 1:
                    num = temp
                    current_unit = tab[current_unit][1]
                else:
                    break
            content.append([elem, str(round(num)) + current_unit])
        return content

    @staticmethod
    def _get_name_sampler(skip_prob=0.5):
        def _mask_name(name):
            surname = name[0]
            masked_part = '*' * (len(name) - 1)
            return surname + masked_part

        while True:
            if random.random() > skip_prob:
                yield _mask_name(mingzi(female_rate=0.5)[0])
            else:
                yield ''

    def _process_med(self, med, case):
        age_dosage_modif_tab = {
            '新生儿期': 0.17, '婴儿期': 0.33, '幼儿期': 0.50, '学龄前期': 0.67, '学龄期': 0.80,
        }
        if case['age'] and case['age'][0] in age_dosage_modif_tab.keys():
            med[1] = str(round(float(med[1]) * age_dosage_modif_tab[case['age'][0]]))
        else:
            med[1] = str(round(float(med[1])))
        return med


    def _prepare_med(self, med):
        med = [[float(x) if i == 1 else x for i, x in enumerate(med_unit)] for med_unit in med]
        return med

    def _add_tokens(self, case: dict[str:Union[list, set]], tokens: Union[list, set, str], place: str,
                    filter_overlapping: bool = False, transfer_elems:bool=True):
        if place not in case.keys():
            case[place] = list()
        if not tokens:
            return case
        if isinstance(tokens, str):
            tokens = [tokens]
        if transfer_elems:
            tokens = set(chain.from_iterable(x.split('&') for x in tokens))
            tokens = set(random.choice(x.split('/')) for x in tokens)
            tokens = set(filter(lambda x: x != '' and x != "''", tokens))
        if filter_overlapping:
            tokens = list(filter(lambda x: x not in case, tokens))
        if isinstance(case[place], list):
            case[place].extend(tokens)
        elif isinstance(case[place], set):
            case[place].update(tokens)
        else:
            raise Exception
        return case

    def _get_case_generator(self):
        while self.num_to_generate:
            case = dict()
            if self.cross_libs_gen:
                self.cross_lib_elems.update(next(self.cross_libs_gen))
            if self.sym_signs_gen is None and self.tong_pul_gen is None:
                logger.warn(f'PSC NUM {self.psc_num}, RULE NUM {self.rule_num} Content is empty. Skip!')
                break
            if self.name_gen:
                name = next(self.name_gen)
                case = self._add_tokens(case, name, 'name', filter_overlapping=False,transfer_elems=False)
            if self.sex_gen:
                sex = next(self.sex_gen)
                case = self._add_tokens(case, sex, 'sex', filter_overlapping=True, transfer_elems=True)
            if self.age_gen:
                age_trait, age_sample = next(self.age_gen)
                case = self._add_tokens(case, age_trait, 'age', filter_overlapping=True, transfer_elems=True)
                case = self._add_tokens(case, age_sample[0][1], 'age_sample', filter_overlapping=False, transfer_elems=False)
            if self.time_of_onset_gen:
                time_of_onset_annotation, time_of_onset_sample = next(self.time_of_onset_gen)
                time_of_onset_sample = '，'.join([''.join(seq) for seq in time_of_onset_sample])
                case = self._add_tokens(case, time_of_onset_annotation, 'time_of_onset', filter_overlapping=True, transfer_elems=True)
                case = self._add_tokens(case, time_of_onset_sample, 'time_of_onset_sample', filter_overlapping=False, transfer_elems=False)
            if self.inducement_gen:
                inducement = next(self.inducement_gen)
                case = self._add_tokens(case, inducement, 'inducement',filter_overlapping=True, transfer_elems=True)
            if self.aggravating_gen:
                aggravating = next(self.aggravating_gen)
                case = self._add_tokens(case, aggravating, 'aggravating', filter_overlapping=True, transfer_elems=True)
            if self.chief_complaint_gen:
                chief_complaint = next(self.chief_complaint_gen)
                case = self._add_tokens(case, chief_complaint, 'sym_phy',filter_overlapping=True, transfer_elems=True)
                case = self._add_tokens(case, chief_complaint, 'chief_complaint',filter_overlapping=True, transfer_elems=True)
            if self.sym_signs_gen:
                sym_signs = next(self.sym_signs_gen)
                case = self._add_tokens(case, sym_signs, 'sym_phy', filter_overlapping=True, transfer_elems=True)
            if self.tong_pul_gen:
                tong_pul = next(self.tong_pul_gen)
                case = self._add_tokens(case, tong_pul, 'tong_pul', filter_overlapping=True, transfer_elems=True)
            if self.pre_treat_gen:
                pre_treat = next(self.pre_treat_gen)
                case = self._add_tokens(case, pre_treat, 'pre_treat', filter_overlapping=True, transfer_elems=True)
            if self.aux_exam_gen:
                aux_exam = next(self.aux_exam_gen)
                case = self._add_tokens(case, aux_exam, 'aux_exam', filter_overlapping=True, transfer_elems=True)
            if self.disease_gen:
                disease = next(self.disease_gen)
                case = self._add_tokens(case, disease, 'disease', filter_overlapping=True, transfer_elems=True)
            if self.past_med_his_gen:
                past_med_his = next(self.past_med_his_gen)
                case = self._add_tokens(case, past_med_his, 'past_med_his', filter_overlapping=True, transfer_elems=True)

            case = {k: list(set(filter(lambda x: x != 'null', v))) for k, v in case.items()}

            if any([case[trait] > max_num for trait, max_num in self.max_num_dict.items()]):
                continue

            if not any([case[place] for place in ['sym_phy', 'tong_pul', 'disease'] if place in case]):  # 若全为空，则跳过
                continue

            if self.add_psc and self.psc:
                case['psc'] = self.psc
            if self.add_med and self.med:
                med = [self._process_med(x, case) for x in copy.deepcopy(self.med)]
                case['med'] = [''.join(x) for x in med]
            if self.add_note and self.note:
                case['note'] = self.note
            if self.add_ref and self.ref:
                case['ref'] = self.ref
            case = {k: v for k, v in case.items() if v}
            case = {k: '，'.join(v) for k, v in case.items()}
            structured_case = [f"{self.item_map[item]}：{case[item]}" for item in self.input_seq if item in case.keys()]
            structured_case = '；'.join(structured_case)
            yield {"structured_case": structured_case,
                   "prescription name": case['psc'],
                   "herbal ingredients and dosage": case['med']
                   }
            self.num_to_generate -= 1

    def get_case_generator_wrapper(self):
        return self._case_generator_wrapper(), self.num_to_generate

    def _case_generator_wrapper(self):
        case_gen = self._get_case_generator()
        for case in case_gen:
            yield case

@dataclass
class CaseGenerator:
    model_name: str
    base_url: str
    api_key: str
    root_elems_path: str
    structured_to_simulated_case_prompt_template_gens: list
    verify_case_prompt: str
    clean_simulated_case_prompt: str
    fast_question_prompt_template_path: str
    reduction_factor: float = field(default=0.1)
    shuffle: bool = field(default=True)
    total_num_to_generate:int = field(default=1)
    structured_case_config: dict = field(default_factory=dict)
    structured_case_path: str = field(default=None)
    num_process: int = field(default=1)
    task_timeout: int = field(default=150)
    verify_case: bool = field(default=True)
    clean_simulated_case: bool = field(default=True)

    def __post_init__(self):
        self.root_elems_list = self._get_root_elem_dict(self.root_elems_path)
        self.psc_ratio_dict = self._get_psc_ratio_dict(self.root_elems_list, self.reduction_factor)
        self.task_name = os.path.splitext(os.path.basename(self.root_elems_path))[0]
        self.save_directory = os.path.join('output_data', self.task_name)
        self.save_intermediate_directory = os.path.join('output_data', self.task_name, 'intermediate')
        os.makedirs(self.save_directory, exist_ok=True)
        os.makedirs(self.save_intermediate_directory, exist_ok=True)
        if self.structured_case_path is None:
            self.structured_case_path = os.path.join(self.save_intermediate_directory, 'structured-simulated.json')

        self.llm_instance = LLMClient(self.model_name, self.base_url, self.api_key)

        with open(self.fast_question_prompt_template_path, 'r', encoding='utf-8') as f:
            self.fast_question_prompt_templates = json.load(f)


    def _get_root_elem_dict(self, TCMRule_path: str):

        df = pd.read_excel(TCMRule_path)
        df = df.fillna('None')
        root_elements_list = []

        def _align_time_unit(x):
            tab = {
                '年': {'年': 1, '月': 12, '天': 365, '小时': 8760},
                '岁': {'岁': 1, '月': 12, '天': 365, '小时': 8760},
                '月': {'月': 1, '天': 30, '小时': 720},
                '天': {'天': 1, '小时': 24},
                '小时': {'小时': 1}
            }
            content = []
            trait, elem_range = x.split('：')
            for sub_elem_range in elem_range.split('，'):
                temp = sub_elem_range.split('@')
                if len(temp) == 1:
                    elem, range = '', temp[-1]
                else:
                    elem, range = temp
                mini, maxi = [pair.split('-') for pair in range.split('->')]
                times = tab[maxi[1]][mini[1]]
                content.append([elem, int(mini[0]), int(maxi[0]) * times, mini[1], maxi[1]])  # 元素 最小值 最大值 最小单位 最大单位
            return trait, content

        def parse_data(strs:Union[list, str]):
            fmt_pattern = r'<([^>]*)>\{([^{}]*)\}'
            cross_lib_pattern = r'((?:<(c[CcOoAa]z?\d+_?\d*)>)+)(.+)'
            elem_dict = dict()
            if isinstance(strs, str):
                strs = [strs]
            for s in strs:
                fmt_matches = re.findall(fmt_pattern, s)
                for fmt_match in fmt_matches:
                    lib_num = fmt_match[0]
                    elems = fmt_match[1]
                    elems = elems.split('；')
                    temp = set()
                    for elem in elems:
                        cross_lib_match = re.search(cross_lib_pattern, elem)
                        if cross_lib_match:
                            cross_libs = re.findall(r'<(c[CcOoAa]z?\d+_?\d*)>', cross_lib_match.group(1))
                            elem = cross_lib_match.group(3)
                            for cross_lib in cross_libs:
                                if cross_lib not in cross_lib_map:
                                    cross_lib_map[cross_lib] = set()
                                cross_lib_map[cross_lib].add(elem)
                        temp.add(elem)
                    elem_dict[lib_num] = temp
            if not elem_dict:
                return elem_dict
            elem_dict = parse_maj(elem_dict)
            return elem_dict

        def parse_maj(elem_dict):
            elem_dict = {k: set(filter(lambda x: x != '' and x != 'null', v)) for k, v in elem_dict.items()}
            elem_dict = {k: v for k, v in elem_dict.items() if len(v) > 0}
            for k, vs in elem_dict.items():
                maj_elms = {v.replace('<maj>', '') for v in vs if '<maj>' in v}
                vs = {v.replace('<maj>', '') for v in vs}
                if len(maj_elms) > 0 and len(maj_elms) < len(vs):
                    elem_dict[k] = {'maj': maj_elms, 'other': vs - maj_elms}
                else:
                    elem_dict[k] = {'maj': vs}
            return elem_dict

        for i, row in df.iterrows():
            cross_lib_map = dict()

            root_elem_dict = dict()
            root_elem_dict['prescription_num'] = row['Prescription No.']

            root_elem_dict['rule_num'] = row['Rule No.']

            root_elem_dict['classification'] = parse_data(row['Category'])

            sym_signs: list[Union[str, list, dict]] = row['Sym and Sign Pool']
            root_elem_dict['sym_signs'] = parse_data(sym_signs)

            sex = row['Gender']
            root_elem_dict['sex'] = parse_data(sex)

            age = parse_data(row['Age'])
            root_elem_dict['age'] = {k: {v_k: [_align_time_unit(v_v) for v_v in v_vs] for v_k, v_vs in v_dict.items()}
                                     for k, v_dict in age.items()}
            assert len(age.keys()) == 1

            # 发病时间元素库 -8
            time_of_onset = row['Onset Time Element Library']
            time_of_onset = parse_data(time_of_onset)
            root_elem_dict['time_of_onset'] = {
                k: {v_k: [_align_time_unit(v_v) for v_v in v_vs] for v_k, v_vs in v_dict.items()}
                for k, v_dict in time_of_onset.items()}
            assert len(time_of_onset.keys()) == 1

            root_elem_dict['aggravating'] = parse_data(row['Condition Aggravation Element Library'])
            root_elem_dict['inducement'] = parse_data(row['Inducement Element Library'])
            root_elem_dict['previous_treatment'] = parse_data(row['Treatment Element Library'])
            root_elem_dict['auxiliary_examination'] = parse_data(row['Auxiliary Examination Element Library'])
            root_elem_dict['diagnosis'] = parse_data(row['Disease Element Library'])
            root_elem_dict['past_med_history'] = parse_data(row['Past History Element Library'])
            root_elem_dict['note'] = row['Notes']
            root_elem_dict['reference'] = row['References']
            tongue_pulse = row['t_color':'p_complex']
            root_elem_dict['tongue_pulse'] = parse_data(tongue_pulse)
            root_elem_dict['prescription'] = row['Prescription Name'].split('；')
            root_elem_dict['medicines'] = [x.split('；') for x in row['Medicine 1':'Medicine 50'] if x != 'None']
            root_elem_dict['cross_libs'] = parse_maj(cross_lib_map)
            root_elements_list.append(root_elem_dict)
        return root_elements_list

    def _get_psc_ratio_dict(self, root_elems_list, reduction_factor):
        psc_ratio_dict = dict()
        for rule in root_elems_list:
            if rule['prescription_num'][:10] not in psc_ratio_dict.keys():
                psc_ratio_dict[rule['prescription_num'][:10]] = 0
            psc_ratio_dict[rule['prescription_num'][:10]] += 1
        ratios = self._adjust_above_mean(list(psc_ratio_dict.values()), reduction_factor)
        for (k, v), n_v in zip(psc_ratio_dict.items(), ratios):
            psc_ratio_dict[k] = n_v / v
        return psc_ratio_dict

    def _adjust_above_mean(self, data, reduction_factor):
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

    def generate_structured_cases(self) -> list:
        container = []
        cnt = 0
        for root_elements in self.root_elems_list:
            if cnt >= self.total_num_to_generate:
                break
            inst = StructuredCase(root_elements, self.psc_ratio_dict,
                                  self.total_num_to_generate, **self.structured_case_config)
            gen, num = inst.get_case_generator_wrapper()
            if num == 0:
                continue
            container.extend([case for case in gen])
            cnt += num
        if self.shuffle:
            random.shuffle(container)
        with open(self.structured_case_path, 'w') as f:
            json.dump(container, f, ensure_ascii=False, indent=2)
        return container

    def generate_syn_case_qa(self, item):

        item['query_history'] = []
        item['response_history'] = []
        item['valid_case_qa_pair'] = {}

        try:
            max_retries = 2
            save_path = os.path.join(self.save_directory, f"{item['process_id']}.json")
            if self.verify_case:
                verify_structured_case_query = self.verify_case_prompt.format(item['structured_case'])
                item['query_history'].append(verify_structured_case_query)
                verify_structured_case_response = self.llm_instance.retry_call(verify_structured_case_query)
                item['response_history'].append(verify_structured_case_response)
                if 'false' in verify_structured_case_response.lower():
                    return 0
            for _ in range(max_retries):
                template_gen = random.choice(self.structured_to_simulated_case_prompt_template_gens)
                task, requirements = next(template_gen)
                head = template_gen.head
                structured_to_simulated_case_query = task + head + item['structured_case'] + requirements
                item['query_history'].append(structured_to_simulated_case_query)
                structured_to_simulated_case_response = self.llm_instance.retry_call(structured_to_simulated_case_query)
                item['response_history'].append(structured_to_simulated_case_response)
                simulated_case = structured_to_simulated_case_response
                prescription = 'Prescription:' + item['prescription name'] + '; ' + item[
                    'herbal ingredients and dosage']

                if self.clean_simulated_case:
                    clean_simulated_case_query = self.clean_simulated_case_prompt.format(
                        simulated_case + '\n' + prescription)
                    item['query_history'].append(clean_simulated_case_query)
                    clean_simulated_case_response = self.llm_instance.retry_call(clean_simulated_case_query)
                    item['response_history'].append(clean_simulated_case_response)
                    simulated_case = clean_simulated_case_response

                if self.verify_case:
                    verify_case_query = self.verify_case_prompt.format(simulated_case + '\n' + prescription)
                    item['query_history'].append(verify_case_query)
                    verify_case_response = self.llm_instance.retry_call(verify_case_query)
                    item['response_history'].append(verify_case_response)
                    if 'false' in verify_case_response.lower():
                        continue
                quick_ques_prompt_template = random.choice(self.fast_question_prompt_templates)
                item['valid_case_qa_pair'] = {
                    "clinical characteristics": quick_ques_prompt_template.format(simulated_case),
                    "prescription name": item['prescription name'],
                    "herbal ingredients and dosage": item['herbal ingredients and dosage']}

            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(item, file, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Failed to process data {item['process_id']} : {str(e)}")
        return 1

    def merge_saved_files(self):
        if not os.path.exists(self.save_directory):
            return []

        json_files = [f for f in os.listdir(self.save_directory) if f.endswith('.json')]
        merged_data = []

        for file in json_files:
            try:
                with open(os.path.join(self.save_directory, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'valid_case_qa_pairs':
                        merged_data.append(data)
            except Exception as e:
                print(f"Error merging file {file}: {e}")
        return merged_data

    def deduplicate_data(self, data, processed_data):
        processed_ids = {item['process_id'] for item in processed_data}
        return [item for item in data if item['process_id'] not in processed_ids]

    def generate(self, restart: bool = False):
        if not os.path.exists(self.structured_case_path) or restart:
            structured_data_list = self.generate_structured_cases()
        else:
            with open(self.structured_case_path, 'r', encoding='utf-8') as file:
                structured_data_list = json.load(file)

        for idx, item in enumerate(structured_data_list, start=1):
            item['process_id'] = idx

        print(f"Loaded {len(structured_data_list)} items.")

        processed_data = self.merge_saved_files()
        print(f"Previously processed items: {len(processed_data)}")

        input_data = self.deduplicate_data(structured_data_list, processed_data)
        print(f"Items remaining for processing: {len(input_data)}")

        with ThreadPoolExecutor(max_workers=self.num_process) as executor:
            future_to_item = {
                executor.submit(
                    self.generate_syn_case_qa,
                    item,
                ): item for item in input_data
            }

            for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Processing Items",
                               unit="item"):
                item = future_to_item[future]
                try:
                    future.result(timeout=self.task_timeout)
                except TimeoutError:
                    print(f"⏰ data {item['process_id']} timeout. Skip! ")
                except Exception as e:
                    print(f"❌ data {item['process_id']} error：{str(e)}")

        final_data = self.merge_saved_files()
        output_path = f"{self.task_name}_final_{len(final_data)}.json"
        print(f"Processed {len(final_data)} items. Saving to {output_path}")

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    model_name = "doubao-seed-2-0-lite-260428"
    base_url="https://ark.cn-beijing.volces.com/api/v3"
    api_key="************************"

    total_num_to_generate = 1
    root_path = os.path.join('tcmdtr', 'tcmdtr.xlsx')

    iptgs = [PromptTemplateGen(**inp_prompt_template_1),
             PromptTemplateGen(**inp_prompt_template_2),
             PromptTemplateGen(**inp_prompt_template_3),
             PromptTemplateGen(**inp_prompt_template_4),]

    fast_question_prompt_template_path = './fast_template.json'
    cg = CaseGenerator(
        model_name,
        base_url,
        api_key,
        root_path,
        iptgs,
        verify_prompt,
        clean_simulated_case_prompt,
        fast_question_prompt_template_path,
        shuffle=True,
        total_num_to_generate=total_num_to_generate,
    )
    cg.generate(restart=True)
