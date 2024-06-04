'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-07-07 11:34:07
LastEditTime: 2023-12-03 11:59:16
Description: 

'''
import json
import os
import re
import shutil
BASE_PATH = "./labeled/"
load_path = BASE_PATH + "abstract.jsonl"
input_data = []
with open(load_path, "r", encoding="utf8") as f:
    for line in f:
        sample = json.loads(line.strip())
        input_data.append(sample)
log_list = []
with open(BASE_PATH + "abstract.log", "r", encoding="utf8") as f:
    for line in f:
        sample = json.loads(line.strip())
        log_list.append(sample)
output_data = []
for i, inp in enumerate(input_data):
    if log_list[i]["labeled"] == 1:
        continue
    inputs = \
"""Assuming you are a professional annotator, your first task is to segment the text into sentences, splitting clauses with the "\n" symbol.
Additionally, after each sentence, if you believe the clause requires cross-modal information, add the "[CM]" symbol at the end of the sentence.
What's more, you need to add the golden answer conclusion in last sentence, if the last sentence do not make any conclusion.

Example:
[[INPUT]]
<QUESTION>\nWhat is the image a representation of?\n\n<CHOICE>\n(A) A dog head\n(B) A cat head\n(C) A bird head\n(D) A rabbit head\n\n<CONTEXT>\n
<ANSWER>\nB\n\n<RATIONALE>\nThe image consists of four tangram shapes; the turquoise part resembles a cat's head, the black part represents the right ear, the lavender part represents the left ear, and the maroon part represents the neck.
[[OUTPUT]]
The image consists of four tangram shapes; [CM]\nthe turquoise part resembles a cat's head, [CM]\nthe black part represents the right ear, [CM]\nthe lavender part represents the left ear, [CM]\nand the maroon part represents the neck. [CM]\nTherefore, the correct answer is b) A cat head.

[[INPUT]]
"""
    inputs += inp["question"] + "\n" + inp["rationale"]
    inputs += "\n[[OUTPUT]]"
    # print(inputs)
    output_data.append({"model": "gpt-3.5-turbo-1106", "messages": [{"role": "user", "content": inputs}], "metadata": {"row_id": i}, "max_tokens": 2048})
with open("labeled-temp/abstract.jsonl", "w", encoding="utf8") as f:
    for out in output_data:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")