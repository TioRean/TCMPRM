import json
import re
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_fixed, stop_after_attempt
from openai import OpenAI

text_to_know_prompt = '''
<text>
{}
</text>

You are a professional medical knowledge extraction expert, specializing in extracting high-value clinical reasoning knowledge points from medical texts. Please extract valid independent medical knowledge points strictly according to the given text, and follow all mandatory rules below:
Extraction Scope
Extract knowledge points with clinical reasoning and training value, covering: syndrome-prescription correspondence relationship, disease clinical manifestations, disease diagnosis, syndrome differentiation criteria, pathogenesis mechanism, prescription (with modification), disease nursing and recuperation schemes, drug pharmacological mechanism, etc.
Elimination Rules
Do not extract trivial descriptive content, repetitive content, empty theoretical discourse, and content without clinical scenario reasoning value.
Knowledge Point Standard
Each extracted knowledge point is independent, complete, self-consistent, no cross-reference and no logical dependency between each other; each knowledge point can independently support the generation of one high-difficulty clinical reasoning question. Do not add any outside knowledge not mentioned in the text.
Output Rules
All extracted knowledge points are written in Chinese. If the text has no valid clinical reasoning knowledge points, directly output an empty array []. Only return a pure JSON array, no extra explanation, no markdown, no redundant symbols. Fixed output format:
```json
["...", "...", ]
```
'''

know_to_qa_prompt = '''
<knowledge>
{}
</knowledge>
Please generate high-difficulty clinical application Q&A pairs based on the given knowledge. Each String in the list generate one or more Q&A pairs if possible. Comply with all the following mandatory rules:
1. Question Rules
Each question must be verifiable, open-ended, independent, and complete. Choose one of suitable question types: clinical case reasoning questions, real clinical scenario-based questions, background comprehensive reasoning questions, or difficult knowledge question. Plain rote memorization is prohibited. Any meaningless or medically valueless questions are not allowed. Do not ask two or more questions once. 
Coverable dimensions include disease condition judgment, diagnosis, syndrome differentiation, prescription selection & modification, nursing and recuperation scheme formulation, pathogenesis, pharmacological mechanism, etc. 
2. Answer criteria
Each question must correspond to **one unique, exclusive and definite standard answer**. The answers should avoid involving two or more entities. Ensure the answers is a short phrase, without any redundant interpretation or extended description.
3. Content restrictions
Q&A pairs must be remain valid without reference to the knowledge. Q&A pairs must be mutually independent; there must be no referential relationship or sequential order between Q&A pairs. Ensure the question is clear and targeted, leaving no room for alternative interpretations or answers from other perspectives. Do not introduce additional knowledge.
4. Output Format Rules
All questions and answers must be written in Chinese.
If a knowledge point is not suitable for constructing Q&A pairs, it should be discarded. If there is no valid clinical reasoning point in the knowledge, directly output an empty array [].
Only return a pure JSON array without any extra text, symbols, notes or markdown content. The fixed output format is:
```json
[{{"question": "xxx", "answer": "xxx"}}]
```
'''

filt_qa_prompt = '''
<Q&A pair>
{}
</Q&A pair>
Please make a judgment without any given background information. Dose the Q&A pair meet any of the following invalid criteria:
1. Be non-verifiable;
2. Contradict common sense or professional medical principles;
3. Contain logical inconsistencies and internal contradictions;
4. Have ambiguous or unclear entity expressions;
5. Be misleading or contain latent potential errors;
6. Lack necessary background information;
7. Have no practical value as professional medical examination questions;
8. Contain format errors; 
9. There are general expressions without a clear reference; any noun or name is not specific or unique;
10. The question lacks pertinence and may be answered from other perspectives;
11. The answer does not address the question;
12. The answer lacks reasoning depth;
13. The answer merely repeats the known information in the question stem.
Do not output any extra explanations, comments, or additional text. Return only "false" if meet the invalid criteria, else "true".'''

know_to_case_psc_prompt = '''
<knowledge>
{}
</knowledge>
Extract all groups of clinical characteristics — TCM prescription mapping data from the given knowledge content. Only extract oral TCM decoction prescriptions. Strictly comply with all the rules below:
1. Write the clinical characteristics in the form of a TCM clinical record. The scope of clinical characteristics includes clinical manifestations, TCM four diagnostic information, predisposing factors, past medical history, auxiliary examination results, disease diagnosis, and other clinically relevant information. The clinical characteristics must contain information for TCM syndrome differentiation and treatment.
2. Each TCM prescription must contain prescription name, herbal ingredients, and specific dosage.
If the given text does not specify the herbal ingredients or dosage, supplement with the classic conventional herbal ingredients and standard clinical fixed dosages of the corresponding classical prescription.
The output prescriptions are final TCM prescriptions in standard writing format. Do not add any extra analysis, reasoning, alternative version listings, prescription modification in the output prescriptions.
The prescription names must be formal, officially recorded classical TCM prescription names (no generalized substitute names allowed). All dosages must adopt specific fixed numerical values, not dosage ranges.
3. Do not extract Chinese patent medicines, external-use TCM formula, rarely-used clinician’s empirical formula, acupuncture and moxibustion therapies, Western medicine medications, and any non-TCM herbal decoction prescriptions, no matter there is a TCM prescription or not. 
4. Each group of mapping data must be independent, complete, logically self-consistent, and fully conform to the given knowledge content when viewed as a standalone entry. Prohibit fabricating any information. 
5. If extracted mapping data are invalid and do not follow all the above rules, discard the mapping data. If no valid mapping data can be extracted, directly output an empty array [].
6. Write in Chinese. Output format requirement:
```json
[{{"clinical characteristics": "xxx", "prescription name": "xxx", "herbal ingredients and dosage": "xxx"}}]
```
'''

filt_case_psc_prompt = '''
<data>
{}
</data>
Judge and output "false" if any of the following situations exist in the data: 
1. Obvious common sense errors;
2. Factual medical and TCM professional knowledge errors;
3. Internal logical contradictions;
4. Rarely used personal empirical exclusive TCM formulas with no universal clinical application value;
5. Prescriptions that violate TCM compatibility contraindications, usage taboos and clinical medication norms; 
6. Lack of information required for TCM syndrome differentiation and treatment
If none of the above situations exist, output "true".
'''

case_to_qa_prompt = '''
<data>
{}
</data>
The data above are clinical characteristics-TCM prescription mapping data. Please reorganize the characteristics into complete medical cases, and then convert these medical cases into **questions** requesting the corresponding TCM prescriptions.
Strict Compliance Requirements:
Retain all medical information, including clinical manifestations, four diagnostic data of TCM, predisposing factors, past medical history, auxiliary examination results, disease diagnosis, and other clinically relevant information. Redundant and irrelevant information can be deleted. The questions must not contain any information that hints at the prescription answers.
Keep the prescription name, herbal ingredients and dosage unchanged with no modification;
Design diverse natural question expressions;
All outputs shall be written in Chinese;
Do not output any extra explanations, remarks or redundant content. Only return the result strictly in the designated JSON format below:
[{{"question": "", "prescription name": "", "herbal ingredients and dosage": ""}}]
'''

filt_case_qa_prompt = """<Q&A pair>
{}
</Q&A pair>
You are a professional TCM medical data quality reviewer. You will receive a TCM clinical case Q&A pair wrapped in <Q&A pair></Q&A pair> tags.
You must strictly follow the rules below and only output exactly one word: true or false, with no explanations, no extra symbols, no additional content of any kind.
Judge and output "false" if any of the following situations exist in the Q&A pair:
1. Obvious common sense errors;
2. Factual medical and TCM professional knowledge errors;
3. Internal logical contradictions;
4. Rarely used personal empirical exclusive TCM formulas with no universal clinical application value;
5. Prescriptions that violate TCM compatibility contraindications, usage taboos and clinical medication norms;
6. The "question" in the Q&A pair is not a valid question or request;
7. The answer is revealed in the question.
If none of the above situations exist, output "true".
"""

class LLMClient:
    def __init__(self, model_name, base_url='http://127.0.0.1:8000/v1', api_key="null"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        print(f"✅Successfully connected to: {self.model_name}")

    def call_llm(self, prompt_content, generation_params=None):
        if generation_params is None:
            generation_params = {
                "max_tokens": 20000,
                "temperature": 0.3,
                "top_p": 0.90,
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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/src/demo/demo_general_verifiable_medical_questions.json", required=False,
                        help="Path to the input JSON data file.")

    parser.add_argument("--model_name", type=str, default="doubao-seed-2-0-lite-260428", help="model name")
    parser.add_argument("--base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3", help="url")
    parser.add_argument("--api_key", type=str, default="**************************", help="api_key")

    parser.add_argument("--num_process", type=int, default=60, help="Number of parallel processes.")
    parser.add_argument("--limit_num", type=int, default=None, help="Limit the number of processed items.")
    parser.add_argument("--generate_know", type=bool, default=True, help="generate knowledge via text")
    parser.add_argument("--generate_know_qa", type=bool, default=True, help="generate general medical questions")
    parser.add_argument("--generate_case_qa", type=bool, default=True, help="generate general questions for prescription recommendation")
    parser.add_argument("--target_text_len", type=int, default=15000,
                        help="Target len of the text in qa generating task.")
    parser.add_argument("--min_text_len", type=int, default=300, help="Min len of the text in qa generating task.")
    parser.add_argument("--filter_know_qa", type=bool, default=True, help="Enable filtering of general medical questions with LLMs.")
    parser.add_argument("--filter_case_qa", type=bool, default=True, help="Enable filtering of general questions for prescription recommendation with LLMs.")
    parser.add_argument("--task_timeout", type=int, default=300,
                        help="Total timeout for single data processing (seconds)")
    return parser.parse_args()


def extract_json_content(text):
    text = re.sub(r'```json|```|`', '', text).strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    return match.group(0).strip() if match else '[]'


def parse_list_str_response(response):
    try:
        json_str = extract_json_content(response)
        parsed_data = json.loads(json_str.replace('\n', ''))
        if not isinstance(parsed_data, list):
            return False, []
        valid_data = [data.strip() for data in parsed_data if isinstance(data, str)]

        if valid_data:
            return True, valid_data
        else:
            return False, []
    except Exception as e:
        print(f"Error parsing response: {e}")
        return False, []


def parse_list_dict_response(response, mod):
    try:
        json_str = extract_json_content(response)
        parsed_data = json.loads(json_str.replace('\n', ''))
        if not isinstance(parsed_data, list):
            return False, []
        if mod == 'know_qa':
            valid_qas = []
            for qa in parsed_data:
                if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                    q = str(qa["question"]).strip()
                    a = str(qa["answer"]).strip()
                    if q and a:
                        valid_qas.append({"question": q, "answer": a})
            if valid_qas:
                return True, valid_qas
            else:
                return False, []
        elif mod == 'case_qa':
            valid_case_qas = []
            for case_qa in parsed_data:
                if isinstance(case_qa, dict) and "question" in case_qa \
                        and "prescription name" in case_qa \
                        and "herbal ingredients and dosage" in case_qa:
                    q = str(case_qa["question"]).strip()
                    pn = str(case_qa["prescription name"]).strip()
                    cd = str(case_qa["herbal ingredients and dosage"]).strip()

                    if q and pn and cd:
                        valid_case_qas.append({"question": q, "prescription name": pn,
                                               "herbal ingredients and dosage": cd})
            if valid_case_qas:
                return True, valid_case_qas
            else:
                return False, []

        elif mod == 'case_psc':
            valid_case_pscs = []
            for case_psc in parsed_data:
                if isinstance(case_psc, dict) and "clinical characteristics" in case_psc \
                        and "prescription name" in case_psc \
                        and "herbal ingredients and dosage" in case_psc:
                    cc = str(case_psc["clinical characteristics"]).strip()
                    pn = str(case_psc["prescription name"]).strip()
                    cd = str(case_psc["herbal ingredients and dosage"]).strip()

                    if cc and pn and cd:
                        valid_case_pscs.append({"question": cc, "prescription name": pn,
                                                "herbal ingredients and dosage": cd})
            if valid_case_pscs:
                return True, valid_case_pscs
            else:
                return False, []

    except Exception as e:
        print(f"Error parsing response: {e}")
        return False, []


def split_and_merge_text(input_str: str, target_text_len: int, min_text_length: int):
    sentence_pattern = re.compile(r'([^.!?。！？]+[.!?。！？])')
    sentence_list = [s.strip() for s in sentence_pattern.findall(input_str) if s.strip()]

    if not sentence_list:
        return []

    result = []
    current_chunk = ""

    for sentence in sentence_list:
        temp = current_chunk + sentence
        if len(temp) <= target_text_len:
            current_chunk = temp
        else:
            if current_chunk:
                result.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        result.append(current_chunk)

    if result and len(result[-1]) < min_text_length:
        result.pop()

    return result


def process_single_item(item, llm_instance, save_directory, text_to_know_prompt, know_to_qa_prompt, filt_qa_prompt,
                        know_to_case_psc_prompt, case_to_qa_prompt, filt_case_psc_prompt, filt_case_qa_prompt,
                        generate_know, generate_know_qa, generate_case_qa, target_text_len, min_text_len,
                        filter_know_qa, filter_case_qa):
    item['know_to_qa_query'] = []
    item['know_to_qa_response'] = []
    item['filt_qa_query'] = []
    item['filt_qa_response'] = []
    item['valid_qa_pairs'] = []

    item['know_to_case_psc_query'] = []
    item['know_to_case_psc_response'] = []
    item['filt_case_psc_query'] = []
    item['filt_case_psc_response'] = []

    item['case_to_qa_query'] = []
    item['case_to_qa_response'] = []
    item['filt_case_qa_query'] = []
    item['filt_case_qa_response'] = []

    item['valid_case_qa_pairs'] = []

    try:
        max_retries = 2
        save_path = os.path.join(save_directory, f"{item['process_id']}.json")
        if generate_know:
            item['text_to_know_query'] = []
            item['text_to_know_response'] = []
            item['valid_know'] = []
            split_text_li = split_and_merge_text(item['text'], target_text_len, min_text_len)
            for split_text in split_text_li:
                text_to_know_query = text_to_know_prompt.format(split_text)
                item['text_to_know_query'].append(text_to_know_query)
                know = ''
                for _ in range(max_retries):
                    text_to_know_response = llm_instance.retry_call(text_to_know_query)
                    item['text_to_know_response'].append(text_to_know_response)
                    valid, text_to_know_parsed_data = parse_list_str_response(text_to_know_response)
                    if valid:
                        know = json.dumps(text_to_know_parsed_data, ensure_ascii=False)
                        break
                item['valid_know'].append(know)
        assert item['valid_know'], "The valid_know list should not be an empty list."
        if generate_know_qa:
            for know in item['valid_know']:
                if not know:
                    continue
                know_to_qa_query = know_to_qa_prompt.format(know)
                item['know_to_qa_query'].append(know_to_qa_query)
                for _ in range(max_retries):
                    response = llm_instance.retry_call(know_to_qa_query)
                    item['know_to_qa_response'].append(response)
                    valid, parsed_data = parse_list_dict_response(response, 'know_qa')
                    if valid:
                        if filter_know_qa:
                            temp = []
                            for qa in parsed_data:
                                filt_qa_query = filt_qa_prompt.format(qa)
                                item['filt_qa_query'].append(filt_qa_query)
                                filt_qa_response = llm_instance.retry_call(filt_qa_query)
                                item['filt_qa_response'].append(filt_qa_response)
                                if 'true' in filt_qa_response.lower():
                                    temp.append(qa)
                            parsed_data = temp
                        item["valid_qa_pairs"].extend(parsed_data)
                        break
        if generate_case_qa:
            for know in item['valid_know']:
                if not know:
                    continue
                know_to_case_psc_query = know_to_case_psc_prompt.format(know)
                item['know_to_case_psc_query'].append(know_to_case_psc_query)
                for _ in range(max_retries):
                    know_to_case_psc_response = llm_instance.retry_call(know_to_case_psc_query)
                    item['know_to_case_psc_response'].append(know_to_case_psc_response)
                    valid, know_to_case_psc_response_parsed_data = parse_list_dict_response(know_to_case_psc_response,
                                                                                            'case_psc')
                    if not valid:
                        continue
                    if filter_case_qa:
                        temp = []
                        for ccpncd in know_to_case_psc_response_parsed_data:
                            filt_case_psc_query = filt_case_psc_prompt.format(ccpncd)
                            item['filt_case_psc_query'].append(filt_case_psc_query)
                            filt_case_psc_response = llm_instance.retry_call(filt_case_psc_query)
                            item['filt_case_psc_response'].append(filt_case_psc_response)
                            if 'true' in filt_case_psc_response.lower():
                                temp.append(ccpncd)

                        know_to_case_psc_response_parsed_data = temp

                    case_pscs = json.dumps(know_to_case_psc_response_parsed_data, ensure_ascii=False)
                    case_to_qa_query = case_to_qa_prompt.format(case_pscs)
                    item['case_to_qa_query'].append(case_to_qa_query)
                    case_to_qa_response = llm_instance.retry_call(case_to_qa_query)
                    valid, case_to_qa_response_parsed_data = parse_list_dict_response(case_to_qa_response, 'case_qa')
                    if not valid:
                        continue
                    if filter_case_qa:
                        temp = []
                        for qpncd in case_to_qa_response_parsed_data:
                            filt_case_qa_query = filt_case_qa_prompt.format(qpncd)
                            item['filt_case_qa_query'].append(filt_case_qa_query)
                            filt_case_qa_response = llm_instance.retry_call(filt_case_qa_query)
                            item['filt_case_qa_response'].append(filt_case_qa_response)
                            if 'true' in filt_case_qa_response.lower():
                                temp.append(qpncd)
                        case_to_qa_response_parsed_data = temp
                    item['valid_case_qa_pairs'].extend(case_to_qa_response_parsed_data)
                    break
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(item, file, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"处理数据 {item['process_id']} 失败: {str(e)}")
    return 1


def merge_saved_files(directory, generate_know, generate_know_qa, generate_case_qa):
    if not os.path.exists(directory):
        return []

    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    merged_data = []

    for file in json_files:
        try:
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                merge_flag = True
                if any([generate_know and not len(data.get('valid_know', [])) > 0,
                        generate_know_qa and not len(data.get('valid_qa_pairs', [])) > 0,
                        generate_case_qa and not len(data.get('valid_case_qa_pairs', [])) > 0]):
                    merge_flag = False
                if merge_flag:
                    merged_data.append(data)
        except Exception as e:
            print(f"Error merging file {file}: {e}")
    return merged_data


def deduplicate_data(data, processed_data):
    processed_ids = {item['process_id'] for item in processed_data}
    return [item for item in data if item['process_id'] not in processed_ids]


def main():
    args = parse_arguments()

    # Load input data
    with open(args.data_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    # Assign unique process IDs to each item
    for idx, item in enumerate(input_data, start=1):
        item['process_id'] = idx

    if args.limit_num:
        input_data = input_data[:args.limit_num]

    print(f"Loaded {len(input_data)} items.")

    # Define task and save directory
    task_name = os.path.splitext(os.path.basename(args.data_path))[0]
    save_directory = os.path.join('output_data', task_name)
    os.makedirs(save_directory, exist_ok=True)

    llm_instance = LLMClient(model_name=args.model_name, base_url=args.base_url, api_key=args.api_key)

    processed_data = merge_saved_files(save_directory, args.generate_know, args.generate_know_qa, args.generate_case_qa)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(input_data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        future_to_item = {
            executor.submit(
                process_single_item,
                item,
                llm_instance,
                save_directory,
                text_to_know_prompt,
                know_to_qa_prompt,
                filt_qa_prompt,
                know_to_case_psc_prompt,
                case_to_qa_prompt,
                filt_case_psc_prompt,
                filt_case_qa_prompt,
                args.generate_know,
                args.generate_know_qa,
                args.generate_case_qa,
                args.target_text_len,
                args.min_text_len,
                args.filter_know_qa,
                args.filter_case_qa,
            ): item
            for item in input_data
        }

        for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Processing Items",
                           unit="item"):
            item = future_to_item[future]
            try:
                # 获取执行结果（超时控制）
                future.result(timeout=args.task_timeout)
            except TimeoutError:
                print(f"⏰ data {item['process_id']} timeout. Skip! ")
            except Exception as e:
                print(f"❌ data {item['process_id']} error：{str(e)}")

    # Merge and save final output
    final_data = merge_saved_files(save_directory, args.generate_know, args.generate_know_qa, args.generate_case_qa)
    output_path = f"{task_name}_final_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
