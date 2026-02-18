# -*- coding: utf-8 -*-
import random
import copy
from dataclasses import dataclass, field
import os
import json
import tempfile



output_template = {
    'head': '医案及处方信息如下：',
    'task': {
        'task_prompts': ['根据给定的中医医案及处方信息，用简洁的语言分析医案的理、法、方、药，回顾及总结相关知识，'
                         '最后写出带有剂量的处方、煎服法及注意事项',]
    },
    'requirements': {
        'basic_requirement_prompts': [
            '不改变原信息含义',
            '用简洁的文字表达',
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


optgs = [PromptTemplateGen(**output_template), ]


def _process_output_resources(file_path):
    try:
        content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                output_template_gen = random.choice(optgs)
                head = output_template_gen.head
                task, requirements = next(output_template_gen)
                data[1] = task + head + data[0] + data[1] + requirements
                content.append([data[0], data[1]])

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         delete=False, suffix='.json') as temp_f:
            for x in content:
                temp_f.write(json.dumps(x, ensure_ascii=False) + '\n')
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


_process_output_resources(os.path.join('Gen', 'processed_simulated.jsonl'))

