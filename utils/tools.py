import json
import os
import time


def read_jsonl(data_path):
    input_data = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                input_data.append(json.loads(line.strip()))
    else:
        print(f"Missing {data_path}")
    return input_data


def write_jsonl(save_path, save_object, mode="a"):
    with open(save_path, mode, encoding="utf8") as f:
        for obj in save_object:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

import openai
openai.api_key = "API-key"
def clp_request(idx, data, source_language, target_language):
    
    instruction = f"Please act as an expert in multi-lingual understanding in {source_language}.\n\n"
    instruction += "Request:\n" + data["text"] + "\n\n"
    instruction += f"Let's understand the task in {target_language} step-by-step!"
    messages = [{"role": "user", "content": instruction}]
    while True:
        try:
            completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo", 
                        messages=messages,
                        temperature=1,
                        top_p=1)
            break
        except Exception as e:
            print(e)
            time.sleep(1)
    messages += [{"role": "assistant", "content":  completion["choices"][0]["message"]["content"]}]
    instruction = f"After understanding, you should act as an expert in arithmetic reasoning in {target_language}.\n"
    instruction += "Let's resolve the task you understand above step-by-step!\n"
    instruction += "Finally, you should format your answer as 'Answer: [num]'."
    messages += [{"role": "user", "content":  instruction}]
    while True:
        try:
            completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo", 
                        messages=messages,
                        temperature=0.2,
                        top_p=1)
            break
        except Exception as e:
            print(e)
            time.sleep(1)
    
    res_list = [x["message"]["content"] for x in completion["choices"]]
    messages = messages + [{"role": "assistant", "content":  res_list[0]}]
    print({"id": idx, "message": messages, "origin": data})
    return {"id": idx, "message": messages, "origin": data}