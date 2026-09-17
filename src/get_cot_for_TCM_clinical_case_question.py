import os
import random
import json
from tqdm import tqdm
import argparse
import re
import traceback
import copy
from concurrent.futures import ThreadPoolExecutor
from retrying import retry
from openai import OpenAI
import openai
import httpx
from concurrent.futures import as_completed

# python -m vllm.entrypoints.openai.api_server --model /root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507/ --served-model-name Qwen3-30B-A3B-Instruct-2507 --port 8000 --trust-remote-code --tensor-parallel-size 4

verify_prompt = """<Model Response>
{}
</Model Response>

<Reference Answer>
{}
</Reference Answer>

Compare the prescription in the model response with the reference answer.
Output **only one** of the following three options, with no extra text, explanation, or analysis:
- "True": if the prescription is exactly identical to the reference.
- "Similar": if the prescription composition, targeted pathogenesis, or therapeutic methods are similar.
- "False": if the prescription is mostly different."""

query_prompt_init = """<question>
{}
</question>

Please answer the above question using the Chain of Thought (CoT) reasoning method.
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
The reasoning process should be divided into multiple steps, each step belongs to one of three types: **"Inner Thinking"**, **"Final Conclusion"**, **"Verification"**.

Requirements for each action:
- **"Inner Thinking"**: In-depth reasoning process. Multiple steps are needed for thorough analysis. Each step MUST include a short, clear title.
- **"Final Conclusion"**: Summarize all previous reasoning and give the final TCM diagnosis and treatment conclusion. No title needed.
- **"Verification"**: Check whether the conclusion is reasonable and consistent with TCM theory. If correct, end; if not, return to Inner Thinking. No title needed.

The output MUST be strictly in the following JSON structure.
ALL content in JSON MUST be written in **Chinese**.
```json
{{
  "CoT": [
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
  ]
}}
```"""

query_prompt_init_w_label = """<question>
{}
</question>

Please answer the above question using the Chain of Thought (CoT) reasoning method.
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Now, I'll secretly tell you that the labeled answer is "{}", but please respond as if you don't know it at all. 
The reasoning process should be divided into multiple steps, each step belongs to one of three types: **"Inner Thinking"**, **"Final Conclusion"**, **"Verification"**.

Requirements for each action:
- **"Inner Thinking"**: In-depth reasoning process. Multiple steps are needed for thorough analysis. Each step MUST include a short, clear title.
- **"Final Conclusion"**: Summarize all previous reasoning and give the final TCM diagnosis or treatment conclusion. No title needed.
- **"Verification"**: Check whether the conclusion is reasonable and consistent with TCM theory. If correct, end; if not, return to Inner Thinking. No title needed.

The output MUST be strictly in the following JSON structure.
ALL content in JSON MUST be written in **Chinese**.
```json
{{
  "CoT": [
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
  ]
}}
```"""

query_prompt_init_w_label_and_remarks = """<question>
{}
</question>

Please answer the above question using the Chain of Thought (CoT) reasoning method.
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Now, I'll secretly tell you that the labeled answer is "{}", with related remarks as follows "{}", but please respond as if you don't know them at all. 
The reasoning process should be divided into multiple steps, each step belongs to one of three types: **"Inner Thinking"**, **"Final Conclusion"**, **"Verification"**.

Requirements for each action:
- **"Inner Thinking"**: In-depth reasoning process. Multiple steps are needed for thorough analysis. Each step MUST include a short, clear title.
- **"Final Conclusion"**: Summarize all previous reasoning and give the final TCM diagnosis or treatment conclusion. No title needed.
- **"Verification"**: Check whether the conclusion is reasonable and consistent with TCM theory. If correct, end; if not, return to Inner Thinking. No title needed.

The output MUST be strictly in the following JSON structure.
ALL content in JSON MUST be written in **Chinese**.
```json
{{
  "CoT": [
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
  ]
}}
```"""

prompt_rethink_Backtracking = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion is not correct**. Your 'Verification' result must be consistent with this judgment. Please use **backtracking** to revisit earlier reasoning nodes, revise and optimize the reasoning process, and derive a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. All content within the JSON fields must be written in **Chinese**. You do not need to repeat previous reasoning. Start directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

prompt_rethink_Backtracking_for_similar = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion is not perfect and requires further refinement**. Your 'Verification' result must be consistent with this judgment. Please use **backtracking** to revisit earlier reasoning nodes, revise and optimize the reasoning process, and derive a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. All content within the JSON fields must be written in **Chinese**. You do not need to repeat previous reasoning. Start directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

prompt_rethink_Exploring_New_Path = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> stands for the question to be answered, and <previous reasoning> contains your earlier reasoning process. Your task is to continue directly from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion is not correct**. Your 'Verification' result must be consistent with this judgment. Please refine the reasoning by exploring new problem-solving approaches, and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. All content within the JSON fields must be written in **Chinese**. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

prompt_rethink_Exploring_New_Path_for_similar = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> stands for the question to be answered, and <previous reasoning> contains your earlier reasoning process. Your task is to continue directly from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion is not perfect and requires further refinement**. Your 'Verification' result must be consistent with this judgment. Please refine the reasoning by exploring new problem-solving approaches, and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. All content within the JSON fields must be written in **Chinese**. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Correction = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion is not correct**. Your 'Verification' result must be consistent with this judgment. Please refine the reasoning by making precise **corrections** to fix the flaws in the prior process, then construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. All content within the JSON fields must be written in **Chinese**. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Correction_for_similar = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion is not perfect and requires further refinement**. Your 'Verification' result must be consistent with this judgment. Please refine the reasoning by making precise **corrections** to fix the flaws in the prior process, then construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. All content within the JSON fields must be written in **Chinese**. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

prompt_w_label = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. Now, I'll secretly tell you that the labeled answer is "{}", but please respond as if you don't know it at all. Your 'Verification' requires careful consideration, and if incorrect, you need to provide new Inner Thinking steps and a new Final Conclusion to ensure the final answer aligns with the correct one.

### Output Format
Strictly follow the JSON structure below. All content within the JSON fields must be written in **Chinese**. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

prompt_w_label_and_remarks = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your answer must strictly follow the logic of **TCM syndrome differentiation and treatment**.
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. Now, I'll secretly tell you that the labeled answer is "{}", with related remarks as follows "{}", but please respond as if you don't know them at all. Your 'Verification' requires careful consideration, and if incorrect, you need to provide new Inner Thinking steps and a new Final Conclusion to ensure the final answer aligns with the correct one.

### Output Format
Strictly follow the JSON structure below. All content within the JSON fields must be written in **Chinese**. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

prompt_reformat_to_cot = """<Thought Process>
{}
</Thought Process>

<Question>
{}
</Question>

The <Thought Process> above reflects the model's reasoning based on the <Question>. Your task is to rewrite the <Thought Process> to resemble a more human-like, intuitive natural thinking process in Chinese. The new version should:

1. Present your reasoning in a step-by-step manner, with each individual thought on a new line separated by a line break.
2. Refrain from using structured titles or formatting; instead, focus on smooth, natural transitions. Use casual and conversational language for transitions or validations—words like "hmm," "oh," "also," or "wait" work well here.
3. Expand the content to make the reasoning more thorough, detailed, and logically coherent, all while keeping the tone conversational and easy to follow.

Return directly the revised natural thinking in JSON format as follows:
```json
{{
  "NaturalReasoning": "..."
}}
```"""

prompt_get_final_response = """<Internal Thinking>
{}
</Internal Thinking>

<Question>
{}
</Question>

The <Internal Thinking> represents your internal thoughts about the <Question>. Now, I'll confidentially inform you that the labeled answer is "{}", but you must act as if unaware. Based on this, craft a rich, high‑quality final response in Chinese. Ensure that your final response closely aligns with the <Question>. Output only the final response, with no additional content."""

search_strategies = [
    ('Backtracking', prompt_rethink_Backtracking),
    ('Exploring New Paths', prompt_rethink_Exploring_New_Path),
    ('Correction', gen_prompt_rethink_Correction)
]

search_strategies_for_similar = [
    ('Backtracking for similar', prompt_rethink_Backtracking_for_similar),
    ('Exploring New Paths for similar', prompt_rethink_Exploring_New_Path_for_similar),
    ('Correction for similar', gen_prompt_rethink_Correction_for_similar)
]

consecutive_failure_count = 0


class LLMClient:
    def __init__(self, model_name, base_url, api_key):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        print(f"✅ Successfully connected to : {self.model_name}")

    def call_llm(self, prompt_content, generation_params=None):
        if generation_params is None:
            generation_params = {
                "max_tokens": 8192,
                "temperature": 0.3,
                "top_p": 0.90,
            }
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_content}],
                # **generation_params,
                timeout=300
            )
            response_content = chat_completion.choices[0].message.content
            return response_content
        except (openai.APITimeoutError, httpx.ReadTimeout):
            print("[Warning] Model call timed out, skipped.")
            llm_raw_response = None
            return llm_raw_response

    @retry(wait_fixed=3000, stop_max_attempt_number=5, retry_on_exception=lambda e: True)
    def call_llm_with_retry(self, prompt_content, generation_params=None):
        return self.call_llm(prompt_content, generation_params)


def extract_json_bracket_content(input_text):
    match_result = re.search(r'\{.*\}', input_text, re.DOTALL)
    return match_result.group(0) if match_result else None


def parse_cot_response(llm_response):
    try:
        if not llm_response:
            return False, None
        if llm_response.strip()[0] != '{':
            llm_response = extract_json_bracket_content(llm_response)
            if not llm_response:
                return False, None
        parsed_data = json.loads(llm_response.replace('\n', ''))
        if "CoT" not in parsed_data or not isinstance(parsed_data["CoT"], list):
            return False, None
        if len(parsed_data["CoT"]) < 3:
            return False, None

        last_three_actions = parsed_data['CoT'][-3:]
        if (last_three_actions[0]['action'] != 'Inner Thinking' or
                last_three_actions[1]['action'] != 'Final Conclusion' or
                last_three_actions[2]['action'] != 'Verification'):
            return False, None

        return True, parsed_data

    except:
        return False, None


def parse_natural_reasoning_response(llm_response):
    try:
        if not llm_response:
            return False, None

        if llm_response.strip()[0] != '{':
            llm_response = extract_json_bracket_content(llm_response)
            if not llm_response:
                return False, None

        parsed_data = json.loads(llm_response.replace('\n', ''))
        if "NaturalReasoning" not in parsed_data or not isinstance(parsed_data["NaturalReasoning"], str):
            return False, None

        return True, parsed_data

    except:
        return False, None


def format_search_stream_to_text(long_cot_list):
    format_template = '### {}\n{}\n'
    result_lines = []

    for cot_item in long_cot_list:
        if 'title' in cot_item:
            result_lines.append(format_template.format(cot_item['title'], cot_item['content']))
        else:
            action_title = cot_item['action'].replace('Final Conclusion', 'Conclusion')
            result_lines.append(format_template.format(action_title, cot_item['content']))

    return '\n'.join(result_lines).strip()


def main():
    arg_parser = argparse.ArgumentParser(description="CoT Reasoning and Search Process Execution Script")
    arg_parser.add_argument("--data_path", type=str, default="/src/tcmdtr_final_1.json", required=False,
                            help="Input data file path")

    arg_parser.add_argument("--model_name", type=str, default="Qwen3-30B-A3B-Instruct-2507", help="model name")
    arg_parser.add_argument("--base_url", type=str, default=None, help="url")
    arg_parser.add_argument("--api_key", type=str, default=None, help="api_key")

    # arg_parser.add_argument("--model_name", type=str, default="doubao-seed-2-0-lite-260428", help="model name")
    # arg_parser.add_argument("--base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3", help="url")
    # arg_parser.add_argument("--api_key", type=str, default="**********************",
    #                         help="api_key")

    arg_parser.add_argument("--max_search_attempts", type=int, default=2, help="Maximum number of search retries")
    arg_parser.add_argument("--max_search_depth", type=int, default=2, help="Maximum search depth")
    arg_parser.add_argument("--efficient_search", type=bool, default=True,
                            help="Whether to enable efficient search strategy")
    arg_parser.add_argument("--init_with_label_ratio", type=float, default=0.6,
                            help="Ratio of initial sampling with labels")
    arg_parser.add_argument("--add_remarks", type=bool, default=False, help="Whether to add remarks")
    arg_parser.add_argument("--num_process", type=int, default=250, help="Number of parallel processing threads")
    arg_parser.add_argument("--limit_num", type=int, default=None, help="Limit the number of data items to process")
    arg_parser.add_argument("--task_timeout", type=int, default=300,
                            help="Total timeout for single data processing (seconds)")

    args = arg_parser.parse_args()

    def filter_valid_dataset(raw_dataset):
        valid_items = []
        for data_item in raw_dataset:
            required_keys = ['Open-ended Verifiable Question', 'Ground-True Answer']
            if all(key in data_item for key in required_keys):
                valid_items.append(data_item)

        print(f"Raw data num: {len(raw_dataset)}. Valid data num: {len(valid_items)}")
        return valid_items

    container = []
    with open(args.data_path, 'r', encoding='utf-8') as file:
        raw_dataset = json.load(file)
        for data_dict in raw_dataset:
            if 'valid_case_qa_pair' in data_dict:
                x = data_dict['valid_case_qa_pair']
                container.append({'Open-ended Verifiable Question': x['clinical characteristics'],
                                   'Ground-True Answer': x['prescription name'] + ', ' + x[
                                       'herbal ingredients and dosage']})

    for data_index, data_item in enumerate(container, 1):
        data_item['process_id'] = data_index

    valid_dataset = filter_valid_dataset(container)
    if args.limit_num:
        valid_dataset = valid_dataset[:args.limit_num]

    data_filename = os.path.splitext(os.path.basename(args.data_path))[0]
    task_name = f'{data_filename}_CoT_search'
    output_root_dir = os.path.join('output_data', task_name)
    os.makedirs(output_root_dir, exist_ok=True)

    llm_client = LLMClient(args.model_name, base_url=args.base_url, api_key=args.api_key)

    def verify(conclusion, answer, data_item):
        try:
            query = verify_prompt.format(conclusion, answer)
            data_item['query_history'].append(query)

            response = llm_client.call_llm_with_retry(query)
            data_item['response_history'].append(response)

            if 'true' in response.lower():
                data_item['verify'].append('True')
                return 'True'
            elif 'similar' in response.lower():
                data_item['verify'].append('Similar')
                return 'Similar'
            else:
                data_item['verify'].append('False')
                return 'False'
        except Exception as e:
            # 验证失败直接返回 False，防止卡死
            data_item['verify'].append('False')
            return 'False'

    def process_single_data_item(data_item):
        try:
            data_item = copy.deepcopy(data_item)
            max_retry_per_step = 2
            process_id = data_item['process_id']

            data_item['Long_CoT'] = []
            data_item['query_history'] = []
            data_item['response_history'] = []
            data_item['struct_response_history'] = []
            data_item['response_type_history'] = []
            data_item['prior_failure_records'] = []
            data_item['verify'] = []

            result_save_path = os.path.join(output_root_dir, f"{process_id}.json")

            init_success = False
            init_with_label = True if random.random() <= args.init_with_label_ratio else False
            if init_with_label:
                if args.add_remarks and data_item['remarks']:
                    init_prompt = query_prompt_init_w_label_and_remarks.format(
                        data_item['Open-ended Verifiable Question'],
                        data_item['Ground-True Answer'],
                        data_item['remarks']
                    )
                else:
                    init_prompt = query_prompt_init_w_label.format(data_item['Open-ended Verifiable Question'],
                                                                   data_item['Ground-True Answer'])
            else:
                init_prompt = query_prompt_init.format(data_item['Open-ended Verifiable Question'])

            data_item['query_history'].append(init_prompt)
            for _ in range(max_retry_per_step):
                llm_raw_response = llm_client.call_llm_with_retry(init_prompt)
                data_item['response_history'].append(llm_raw_response)
                init_success, parsed_struct = parse_cot_response(llm_raw_response)
                if init_success:
                    data_item['struct_response_history'].append(parsed_struct["CoT"])
                    data_item['Long_CoT'] = parsed_struct["CoT"]
                    if init_with_label:
                        data_item['response_type_history'].append('Init_CoT_with_label')
                    else:
                        data_item['response_type_history'].append('Init_CoT')
                    break

            if not init_success:
                raise Exception("Initialization of CoT parsing failed.")

            if init_with_label:
                data_item['verify'].append('True')
            else:
                verify(data_item['Long_CoT'][-2]['content'], data_item['Ground-True Answer'], data_item)

            initial_snapshot = copy.deepcopy(data_item)

            if not init_with_label:
                for search_round in range(args.max_search_attempts):
                    if search_round > 0:
                        data_item = copy.deepcopy(initial_snapshot)

                    for depth_step in range(random.randint(1, args.max_search_depth)):
                        if data_item['verify'][-1] == 'True':
                            break
                        reasoning_context = json.dumps(data_item['Long_CoT'][:-1], ensure_ascii=False)
                        if data_item['verify'][-1] == 'False':
                            if depth_step == 0:
                                strategy_name, strategy_prompt = random.choice(search_strategies[1:])
                            else:
                                strategy_name, strategy_prompt = random.choice(search_strategies)
                        elif data_item['verify'][-1] == 'Similar':
                            if depth_step == 0:
                                strategy_name, strategy_prompt = random.choice(search_strategies_for_similar[1:])
                            else:
                                strategy_name, strategy_prompt = random.choice(search_strategies_for_similar)
                        else:
                            raise Exception
                        rethink_prompt = strategy_prompt.format(
                            data_item['Open-ended Verifiable Question'],
                            reasoning_context
                        )
                        data_item['query_history'].append(rethink_prompt)
                        rethink_success = False

                        for _ in range(max_retry_per_step):
                            rethink_response = llm_client.call_llm_with_retry(rethink_prompt)
                            rethink_success, parsed_rethink = parse_cot_response(rethink_response)
                            if rethink_success:
                                data_item['response_history'].append(rethink_response)
                                data_item['struct_response_history'].append(parsed_rethink["CoT"])
                                data_item['Long_CoT'] = data_item['Long_CoT'][:-1] + parsed_rethink["CoT"]
                                data_item['response_type_history'].append(f'Re_CoT_{strategy_name}')
                                break

                        if not rethink_success:
                            raise Exception(f"Depth {depth_step} rethink CoT failed")
                        verify(data_item['Long_CoT'][-2]['content'], data_item['Ground-True Answer'], data_item)

                    if data_item['verify'][-1] == 'True':
                        break

                if data_item['verify'][-1] != 'True' and args.efficient_search:
                    reasoning_context = json.dumps(data_item['Long_CoT'][:-1], ensure_ascii=False)
                    if args.add_remarks:
                        label_guide_prompt = prompt_w_label_and_remarks.format(
                            data_item['Open-ended Verifiable Question'],
                            reasoning_context,
                            data_item['Ground-True Answer'],
                            data_item['remarks']
                        )
                    else:
                        label_guide_prompt = prompt_w_label.format(
                            data_item['Open-ended Verifiable Question'],
                            reasoning_context,
                            data_item['Ground-True Answer']
                        )
                    data_item['query_history'].append(label_guide_prompt)
                    label_success = False

                    for _ in range(max_retry_per_step):
                        label_response = llm_client.call_llm_with_retry(label_guide_prompt)
                        label_success, parsed_label = parse_cot_response(label_response)
                        if label_success:
                            data_item['response_history'].append(label_response)
                            data_item['struct_response_history'].append(parsed_label["CoT"])
                            data_item['Long_CoT'] = data_item['Long_CoT'][:-1] + parsed_label["CoT"]
                            data_item['response_type_history'].append('Label_CoT')
                            data_item['verify'].append('True')
                            break

                    if not label_success:
                        raise Exception("Label-guided optimization failed")

            if data_item['verify'][-1] == 'True':
                search_stream_text = format_search_stream_to_text(data_item['Long_CoT'])
                reformat_prompt = prompt_reformat_to_cot.format(
                    search_stream_text,
                    data_item['Open-ended Verifiable Question']
                )
                data_item['query_history'].append(reformat_prompt)
                reformat_success = False

                for _ in range(max_retry_per_step):
                    nat_response = llm_client.call_llm_with_retry(reformat_prompt)
                    reformat_success, parsed_nat = parse_natural_reasoning_response(nat_response)
                    if reformat_success:
                        data_item['response_history'].append(nat_response)
                        data_item["Complex_CoT"] = parsed_nat["NaturalReasoning"]

                        final_prompt = prompt_get_final_response.format(
                            data_item["Complex_CoT"],
                            data_item['Open-ended Verifiable Question'],
                            data_item['Ground-True Answer']
                        )
                        data_item['query_history'].append(final_prompt)
                        final_response = llm_client.call_llm_with_retry(final_prompt)
                        data_item['response_history'].append(final_response)
                        data_item["Final_Response"] = final_response
                        data_item['Question'] = data_item['Open-ended Verifiable Question']
                        break

                if not reformat_success:
                    raise Exception("Naturalization of CoT formatting unsuccessful.")

            with open(result_save_path, 'w', encoding='utf-8') as f:
                json.dump(data_item, f, ensure_ascii=False, indent=2)
        except Exception as e:
            process_id = data_item.get('process_id', 'unknown')
            print(f"⚠️  task {process_id} error")
            traceback.print_exc()
        return 1

    def collect_all_successful_results(output_dir):
        all_valid_results = []
        for filename in os.listdir(output_dir):
            if not filename.endswith('.json'):
                continue
            try:
                file_path = os.path.join(output_dir, filename)
                with open(file_path, encoding='utf-8') as f:
                    result_item = json.load(f)
                if 'Complex_CoT' in result_item and 'Final_Response' in result_item:
                    all_valid_results.append(result_item)
            except:
                continue
        return all_valid_results

    finished_results = collect_all_successful_results(output_root_dir)
    finished_process_ids = {item['process_id'] for item in finished_results}
    todo_dataset = [item for item in valid_dataset if item['process_id'] not in finished_process_ids]

    print(f"Processing completed: {len(finished_results)} items, Pending: {len(todo_dataset)} items")

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        future_to_item = {
            executor.submit(process_single_data_item, item): item
            for item in todo_dataset
        }

        for future in tqdm(as_completed(future_to_item), total=len(future_to_item)):
            item = future_to_item[future]
            try:
                future.result(timeout=args.task_timeout)
            except TimeoutError:
                print(f"⏰ data {item['process_id']} timeout. Skip! ")
            except Exception as e:
                print(f"❌ data {item['process_id']} error：{str(e)}")

    final_all_results = collect_all_successful_results(output_root_dir)
    final_output_path = f"{task_name}_{len(final_all_results)}.json"
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_all_results, f, ensure_ascii=False, indent=2)

    print(
        f"✅ All processing completed! Final file: {final_output_path}, total {len(final_all_results)} valid data entries")


if __name__ == '__main__':
    main()
