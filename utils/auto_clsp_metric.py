import os
import re

import numpy as np

from utils.tools import read_jsonl

def get_pred_max(data_list):
    pred_max = {}
    params = data_list["params"]
    for j, (key, weight) in enumerate([(x["language"], x["alignment score"]) for x in params]):
        data = data_list[key]
        mes = data["message"][-1]["content"].split("Answer:")[-1].replace(",", "")
        pred_list = [s for s in re.findall(r'-?\d+\.?\d*', mes)]
        if len(pred_list)==0:
            continue
        pred = str(float(pred_list[-1]))
        if pred not in pred_max:
            pred_max[pred] = []
        pred_max[pred].append({"source": j, "value": pred, "weight": weight}) # weight
    return pred_max

def extract_max(pred_max):
    max_data = 0
    max_index = 0
    for x in pred_max:
        weight_sum = sum([t["weight"] for t in pred_max[x]])
        if weight_sum>max_data:
            max_data = weight_sum
            max_index = x
    return max_index, max_data

def judge_equal(pred, answer):
    flag = (np.array([int(float(pred)*100)]) == np.array([int(float(answer)*100)])).sum().item()
    return flag == 1

def compute_result(output_data, lang, mode="clsp"):
    data_all = []
    total = 0
    correct = 0
    for idx, data_list in enumerate(output_data.input_data[lang]):
        total += 1
        pred_max = get_pred_max(data_list)
        pred, _ = extract_max(pred_max)
        answer = data_list["en"]["origin"]["answer"].replace(",", "")
        
        if judge_equal(pred, answer):
            correct += 1
        
    if total == 0:
        return 0, 0
    return correct*100.0/total, total