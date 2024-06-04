'''
Author: Qiguang Chen
LastEditors: Yongheng Zhang
Date: 2023-05-22 17:58:08
LastEditTime: 2024-6-4 16:47:56
Description: 
'''
import fire
from utils.auto_clsp_metric import compute_result as mm
from utils.choice_metric import compute_result as cm
from prettytable import PrettyTable

from utils.datasets import MGSMAutoCLSPOutput


DATA_DICT = {
    "mgsm": {
        "LANG_DICT": {
            "bn": "Bengali", 
            "de": "German",
            "es": "Spanish",
            "fr": "French",
            "ja": "Japanese",
            "ru": "Russian",
            "sw": "Swahili",
            "te": "Telugu",
            "th": "Thai",
            "zh": "Chinese"
        },
        "AUTO-CLSP": {
            "data_path": "mgsm/output/clsp",
            "metric_mode": "auto-clsp"
        },
        
    },
    "paws-x": {
        "LANG_DICT": {
            "de": "German",
            "es": "Spanish",
            "fr": "French",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese"
        },
        "CLP": {
            "data_path": "paws-x/output/clp",
            "metric_mode": "common"
        },
        "en-cot": {
            "data_path": "paws-x/output/en-cot",
            "metric_mode": "common"
        },
    },
    "xnli": {
        "LANG_DICT": {
            "ar": "Arabic",
            "bg": "Bulgarian",
            "de": "German",
            "el": "Greek",
            "es": "Spanish",
            "fr": "French",
            "hi": "Hindi",
            "ru": "Russian",
            "sw": "Swahili",
            "th": "Thai",
            "tr": "Turkish",
            "ur": "Urdu",
            "vi": "Vietnamese",
            "zh": "Chinese",
        },
        "CLP": {
            "data_path": "xnli/output/clp",
            "metric_mode": "common"
        },
        "en-cot": {
            "data_path": "xnli/output/en-cot",
            "metric_mode": "common"
        },
    },
    "xcopa": {
        "LANG_DICT": {
            "et": "Estonian",
            "ht": "Haitian",
            "id": "Indonesian",
            "it": "Italian",
            "qu": "Southern",
            "sw": "Swahili",
            "ta": "Tamil",
            "th": "Thai",
            "tr": "Turkish",
            "vi": "Vietnamese",
            "zh": "Chinese",
        },
        "CLP": {
            "data_path": "xcopa/output/clp",
            "metric_mode": "common"
        },
        "CLSP": {
            "data_path": "xcopa/output/clsp",
            "metric_mode": "clsp"
        },
    }
    
}

SELECTION_PATH = "auto-clsp-exp/mgsm/l6-final-tp02-tp0/output-2"

def main(
    dataset_name="mgsm",
    exp_name="AUTO-CLSP",
):
    exp = DATA_DICT[dataset_name][exp_name]
    LANG_DICT = DATA_DICT[dataset_name]["LANG_DICT"]
    acc = 0
    data_path = exp["data_path"]
    table = PrettyTable(["Language", "Acc", "Total"])
    if dataset_name == "mgsm":
        compute_fn = mm
    else:
        compute_fn = cm # 还没适配
    
    output_data = MGSMAutoCLSPOutput()
    output_data.load_from_merge(SELECTION_PATH, data_path)
    for lang in LANG_DICT.keys():
        accuracy, total = compute_fn(output_data, lang, mode=exp["metric_mode"])
        acc += accuracy
        table.add_row([lang, round(accuracy, 1), total])
    
    table.add_row(["AVG", round(acc/len(LANG_DICT), 1), "-"])
    print(table)

if __name__ == '__main__':
    fire.Fire(main)