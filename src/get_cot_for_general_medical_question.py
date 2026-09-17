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
import argparse
import copy
import json
import os
import random
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import httpx
from openai import OpenAI
from openai import APITimeoutError
import httpx
from openai import OpenAI
from openai import APITimeoutError
from tqdm import tqdm

verify_prompt = """<Model Response>  
{}  
</Model Response>  

<Reference Answer>  
{}
</Reference Answer>  

You are provided with a model-generated response (<Model Response>) and a reference answer (<Reference Answer>). Compare the model response with the reference answer and determine its correctness. Your task is to simply output "True" if the response is correct, and "False" otherwise."""

query_prompt_init = """<question>
{}
</question>

Please respond to the above question <question> using the Chain of Thought (CoT) reasoning method. Your response should consist of multiple steps, each of which includes three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

- **'Inner Thinking'**: This is the step where thinking is done. Note that multiple 'Inner Thinking' steps are required to describe thorough reasoning. Each step should first generate a brief title.
- **'Final Conclusion'**: At this stage, you summarize the correct reasoning from previous 'Inner Thinking' steps and provide the final answer. No title is required here.
- **'Verification'**: At this stage, you verify the conclusion from the "Final Conclusion" step. If the conclusion holds, end the process. If not, return to "Inner Thinking" for further reasoning. No title is required here.

The output format must strictly follow the JSON structure below, and all content within the JSON fields should be written in **Chinese**:
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

gen_prompt_rethink_Backtracking = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning using **backtracking** to revisit earlier points of reasoning and construct a new Final Conclusion.

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

gen_prompt_rethink_Exploring_New_Path = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by exploring new approaches to solving this problem and construct a new Final Conclusion.

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
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by making precise **corrections** to address prior flaws and construct a new Final Conclusion.

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

reformat_to_complex_cot_prompt = """<Thought Process>
{}
</Thought Process>

<Question>
{}
</Question>

The <Thought Process> above reflects the model's reasoning based on the <Question>. Your task is to rewrite the <Thought Process> to resemble a more human-like, intuitive natural thinking process in Chinese. The new version should:

1. Be presented as step-by-step reasoning, with each thought on a new line separated by a line break.
2. Avoid structured titles or formatting, focusing on natural transitions. Use casual and natural language for transitions or validations, such as "hmm," "oh," "also," or "wait."
4. Expand the content, making the reasoning richer, more detailed, and logically clear while still being conversational and intuitive.

Return directly the revised natural thinking in JSON format as follows:
```json
{{
  "NaturalReasoning": "..."
}}
```"""

get_final_response_prompt = """<Internal Thinking>
{}
</Internal Thinking>

<Question>
{}
</Question>

The <Internal Thinking> represents your internal thoughts about the <Question>. Based on this, generate a rich and high-quality final response to the user in Chinese. If there is a clear answer, provide it first. Ensure your final response closely follows the <Question>. Output only your final response, without any additional content."""

# ===================== Configuration & Strategy Definition =====================
SEARCH_STRATEGIES = [
    ("Backtracking", gen_prompt_rethink_Backtracking),
    ("Exploring New Paths", gen_prompt_rethink_Exploring_New_Path),
    ("Correction", gen_prompt_rethink_Correction),
]


# ===================== LLM Client Wrapper =====================
class LLMClient:
    def __init__(self, model_name, base_url="http://127.0.0.1:8000/v1", api_key="null"):
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


# ===================== Text & Response Parsing Utilities =====================
def extract_bracket_content(text: str):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None


def parse_llm_response(response: str):
    try:
        if response[0] != '{':
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n', ''))
        assert isinstance(da["CoT"], list), "CoT should be list"
        assert da['CoT'][-3]['action'] == 'Inner Thinking', 'Inner Thinking should be the third last action'
        assert da['CoT'][-2]['action'] == 'Final Conclusion', 'Final Conclusion should be the second last action'
        assert da['CoT'][-1]['action'] == 'Verification', 'Verification should be the last action'
        return True, da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False, None


def parse_llm_response_reformat(response: str):
    try:
        if response[0] != '{':
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n', ''))
        assert isinstance(da["NaturalReasoning"], str), "NaturalReasoning should be str"
        assert '\n' in da["NaturalReasoning"], r"NaturalReasoning should have \n"
        return True, da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False, None


def get_stream_of_search(longcot: list):
    temp_fmt = "### {}\n{}\n"
    resstr = []
    for x in longcot:
        if 'title' in x:
            resstr.append(temp_fmt.format(x['title'], x['content']))
        else:
            act_name = x['action'].replace('Final Conclusion', 'Conclusion')
            resstr.append(temp_fmt.format(act_name, x['content']))
    return '\n'.join(resstr).strip()


# ===================== Data Helper Functions =====================
def filter_data(tmpdata: list):
    filtered_data = []
    for item in tmpdata:
        if "Open-ended Verifiable Question" not in item or "Ground-True Answer" not in item:
            continue
        filtered_data.append(item)
    print(f"Original data size: {len(tmpdata)}, Filtered data size: {len(filtered_data)}")
    return filtered_data


def deduplicate_data(data: list, processed_data: list):
    processed_ids = {item['process_id'] for item in processed_data}
    return [item for item in data if item['process_id'] not in processed_ids]


def merge_saved_files(save_dir: str):
    res = []
    for root, _, filenames in os.walk(save_dir):
        json_files = [f for f in filenames if f.endswith(".json")]
        for fname in json_files:
            fp = os.path.join(root, fname)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    da = json.load(f)
                    assert "Complex_CoT" in da and "Response" in da
                    res.append(da)
            except Exception:
                continue
        break
    return res


# ===================== Main Processing Logic =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/src/demo_general_verifiable_medical_questions_final_1.json",
                        required=False, help="Path to the input JSON data file.")
    # parser.add_argument("--model_name", type=str, default="doubao-seed-2-0-lite-260428", help="model name")
    # parser.add_argument("--base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3", help="url")
    # parser.add_argument("--api_key", type=str, default="*************************", help="api_key")

    parser.add_argument("--model_name", type=str, default="Qwen3-30B-A3B-Instruct-2507", help="model name")
    parser.add_argument("--base_url", type=str, default=None, help="url")
    parser.add_argument("--api_key", type=str, default=None, help="api_key")
    parser.add_argument("--max_search_attempts", type=int, default=2, help="Maximum number of search attempts.")
    parser.add_argument("--max_search_depth", type=int, default=2, help="Maximum search depth.")
    parser.add_argument("--num_process", type=int, default=250, help="Number of parallel processes.")
    parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")
    parser.add_argument("--task_timeout", type=int, default=300,
                        help="Total timeout for single data processing (seconds)")

    args = parser.parse_args()

    container = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        tmpdata = json.load(f)
        for data_dict in tmpdata:
            if 'valid_qa_pairs' in data_dict:
                container.extend([{'Open-ended Verifiable Question': x['question'],
                                   'Ground-True Answer': x['answer']} for x in data_dict['valid_qa_pairs']])

    proc_id_counter = 1
    for item in container:
        item["process_id"] = proc_id_counter
        proc_id_counter += 1

    data = filter_data(container)
    if args.limit_num is not None:
        data = data[:args.limit_num]
    print(f"read data:{len(data)}")

    task_name = f"{os.path.split(args.data_path)[-1].replace('.json', '')}_CoT_search"
    save_dir = os.path.join("output_data", task_name)
    os.makedirs(save_dir, exist_ok=True)

    llm_instance = LLMClient(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key
    )

    global wrongtime
    wrongtime = 0

    def verify_llm(conclusion: str, answer: str, d: dict):
        query = verify_prompt.format(conclusion, answer)
        response = llm_instance.call_llm_with_retry(query)
        d['llm_query_cot'].append(query)
        d['llm_response_cot'].append(response)
        if "true" in (response or "").lower():
            d['verify'].append(True)
            return True
        d['verify'].append(False)
        return False

    def write_piece_order_data(d: dict):
        global wrongtime
        retry_time = 1
        d['verify'] = []
        d['Long_CoT'] = []
        d['llm_query_cot'] = []
        d['llm_response_cot'] = []
        d['response_struct'] = []
        d['response_type'] = []
        d['prior_fail_try'] = []

        save_path = os.path.join(save_dir, f"{d['process_id']}.json")

        # Initial CoT generation
        query = query_prompt_init.format(d['Open-ended Verifiable Question'])
        d['llm_query_cot'].append(query)
        flag = False
        struct = None
        for _ in range(retry_time):
            response = llm_instance.call_llm_with_retry(query)
            d['llm_response_cot'].append(response)
            flag, struct = parse_llm_response(response)
            if flag:
                d['response_struct'].append(struct["CoT"])
                d['Long_CoT'] = struct["CoT"]
                d['response_type'].append('Init_CoT')
                break
            print(f'retrying Init_CoT', flush=True)
        if not flag:
            raise Exception('init error')

        verify_llm(d['Long_CoT'][-2]['content'], d['Ground-True Answer'], d)

        # Search attempts loop
        for rethinking_try_time in range(args.max_search_attempts):
            if rethinking_try_time > 0:
                del d['prior_fail_try']
                save_d['prior_fail_try'].append(d)
                d = save_d
            save_d = copy.deepcopy(d)

            for rethink_time in range(args.max_search_depth):
                if d['verify'][-1]:
                    break
                reasoning = json.dumps(d['Long_CoT'][:-1], ensure_ascii=False, indent=2)
                if rethink_time > 0:
                    strategy_name, strategy_func = random.choice(SEARCH_STRATEGIES)
                else:
                    strategy_name, strategy_func = random.choice(SEARCH_STRATEGIES[1:])

                query = strategy_func.format(d['Open-ended Verifiable Question'], reasoning)
                d['llm_query_cot'].append(query)
                flag = False
                struct = None
                for _ in range(retry_time):
                    response = llm_instance.call_llm_with_retry(query)
                    flag, struct = parse_llm_response(response)
                    if flag:
                        d['llm_response_cot'].append(response)
                        d['response_struct'].append(struct["CoT"])
                        d['Long_CoT'] = d['Long_CoT'][:-1] + struct["CoT"]
                        d['response_type'].append(f'Re_CoT_{strategy_name}')
                        break
                    print(f'retrying strategy {strategy_name}', flush=True)
                if not flag:
                    raise Exception('rethink error')
                verify_llm(d['Long_CoT'][-2]['content'], d['Ground-True Answer'], d)
            if d['verify'][-1]:
                break

        # Generate Complex_CoT and final response
        if d['verify'][-1]:
            sos = get_stream_of_search(d['Long_CoT'])
            query = reformat_to_complex_cot_prompt.format(sos, d['Open-ended Verifiable Question'])
            d['llm_query_cot'].append(query)
            flag = False
            struct = None
            for _ in range(retry_time):
                response = llm_instance.call_llm_with_retry(query)
                flag, struct = parse_llm_response_reformat(response)
                if flag:
                    d['llm_response_cot'].append(response)
                    d["Complex_CoT"] = struct["NaturalReasoning"]
                    query_final = get_final_response_prompt.format(d['Complex_CoT'],
                                                                   d['Open-ended Verifiable Question'])
                    d['llm_query_cot'].append(query_final)
                    resp_final = llm_instance.call_llm_with_retry(query_final)
                    d['llm_response_cot'].append(resp_final)
                    d["Response"] = resp_final
                    d["Question"] = d['Open-ended Verifiable Question']
                    break

        with open(save_path, mode="w", encoding="utf-8") as fw:
            json.dump(d, fw, ensure_ascii=False, indent=2)
        wrongtime = 0
        return 1

    # Resume from existing outputs
    processed_data = merge_saved_files(save_dir)
    print(f"Previously processed items: {len(processed_data)}")
    data = deduplicate_data(data, processed_data)
    print(f"Items remaining for processing: {len(data)}")

    # Thread pool execution
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        future_to_item = {
            executor.submit(write_piece_order_data, item): item
            for item in data
        }
        for future in tqdm(as_completed(future_to_item), total=len(future_to_item)):
            item = future_to_item[future]
            try:
                future.result(timeout=args.task_timeout)
            except TimeoutError:
                print(f"⏰ data {item['process_id']} timeout. Skip! ")
            except Exception as e:
                print(f"❌ data {item['process_id']} error：{str(e)}")


if __name__ == "__main__":
    main()
