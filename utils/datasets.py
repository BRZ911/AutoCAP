'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-12-03 10:38:43
LastEditTime: 2023-12-21 00:17:09
Description: 

'''

# class RequestData():
#     def __init__(self, model="gpt-3.5-turbo", max_tokens=2048) -> None:
#         self.data = {"model": model, "messages": [{"role": "user", "content": inputs}], "metadata": {"row_id": i}, : 2048})

import json
import os

from tqdm import tqdm
from utils.prompt import LanguageChoicePrompt, TwoStageLanguageChoicePrompt
from utils.tools import clp_request, read_jsonl, write_jsonl


class MGSM():
    def __init__(self, load_dir_path) -> None:
        input_data = {}
        self.LANG_DICT= {
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
        }
        for lang in self.LANG_DICT:
            input_data[lang] = []
            with open(os.path.join(load_dir_path, f"mgsm_{lang}.tsv"), "r", encoding="utf8") as f:
                for i, line in enumerate(f):
                    data = line.strip().split("\t")
                    input_data[lang].append({"id": i, "text": data[0], "answer": data[1]})
                
        self.input_data = input_data
        self.lcp = TwoStageLanguageChoicePrompt()
            
    def parse_language_choice_request_stage_1(self, save_dir_path, temperature=0.2):
        self.save_dir_path=save_dir_path
        for lang in self.LANG_DICT:
            output_data = [
                {"model": "gpt-3.5-turbo-1106",
                "messages": [{"role": "user", "content": self.lcp.generate_prompt(self.LANG_DICT[lang], x["text"], step=0)}],
                "metadata": {"row_id": x["id"]},
                "temperature": temperature}
                for x in self.input_data[lang]
            ]
            if not os.path.exists(os.path.join(save_dir_path, "input")):
                os.makedirs(os.path.join(save_dir_path, "input"),exist_ok=True)
            write_jsonl(os.path.join(save_dir_path, "input", f"request_{lang}.jsonl"), output_data)
    
    def parse_language_choice_request_stage_2(self, load_dir_path, save_dir_path, temperature=0.2):
        self.save_dir_path=save_dir_path
        for lang in self.LANG_DICT:
            last_output = ["" for x in self.input_data[lang]]
            with open(os.path.join(load_dir_path, f"request_{lang}_res.jsonl"), "r", encoding="utf8") as f:
                for i, line1 in enumerate(tqdm(f)):
                    respond_txt = json.loads(line1.strip())[1]["choices"][0]["message"]["content"]
                    idx = json.loads(line1.strip())[-1]["row_id"]
                    last_output[idx] = respond_txt
            output_data = [
                {"model": "gpt-3.5-turbo-1106",
                "messages": [
                    {"role": "user", "content": self.lcp.generate_prompt(self.LANG_DICT[lang], x["text"], step=0)},
                    {"role": "assistant", "content": y},
                    {"role": "user", "content": self.lcp.generate_prompt(self.LANG_DICT[lang], x["text"], step=1)},
                ],
                "metadata": {"row_id": x["id"]},
                "temperature": temperature}
                for x, y in zip(self.input_data[lang], last_output)
            ]
            if not os.path.exists(os.path.join(save_dir_path, "input-2")):
                os.makedirs(os.path.join(save_dir_path, "input-2"),exist_ok=True)
            write_jsonl(os.path.join(save_dir_path, "input-2", f"request_{lang}.jsonl"), output_data)
    
    def request_loop(self, step=0):
        if step == 0:
            for lang in self.LANG_DICT:
                if not os.path.exists(os.path.join(self.save_dir_path, "output")):
                    os.makedirs(os.path.join(self.save_dir_path, "output"),exist_ok=True)
                os.system('python request.py '
                f'--requests_filepath ' + os.path.join(self.save_dir_path, "input", f"request_{lang}.jsonl") + " "
                '--save_filepath '  + os.path.join(self.save_dir_path, "output", f"request_{lang}_res.jsonl") + " "
                '--max_attempts 5 '
                '--logging_level 20')
        else:
            for lang in self.LANG_DICT:
                if not os.path.exists(os.path.join(self.save_dir_path, "output-2")):
                    os.makedirs(os.path.join(self.save_dir_path, "output-2"),exist_ok=True)
                os.system('python request.py '
                f'--requests_filepath ' + os.path.join(self.save_dir_path, "input-2", f"request_{lang}.jsonl") + " "
                '--save_filepath '  + os.path.join(self.save_dir_path, "output-2", f"request_{lang}_res.jsonl") + " "
                '--max_attempts 5 '
                '--logging_level 20')

class MGSMAutoCLSPOutput():
    def __init__(self) -> None:
        self.LANG_DICT= {
            "bn": "Bengali", 
            "de": "German",
            "es": "Spanish",
            "fr": "French",
            "ja": "Japanese",
            "ru": "Russian",
            "sw": "Swahili",
            "te": "Telugu",
            "th": "Thai",
            "zh": "Chinese",
        }
        self.reverse_dict = {y: x for x,y in self.LANG_DICT.items()}
        self.reverse_dict["English"] = "en"
    
    def extract_languages(self,
                          i,
                          respond_txt,
                          APPEND_DICT,
                          lang,
                          input_data,
                          mgsm_data
                          ):
        temp_list = []
        more_data = {}
        request_flag = False
        for param_list_txt in respond_txt.split("=")[-1].split("}"):
            if "{" not in param_list_txt:
                continue
            param_list_txt = param_list_txt.strip(",").strip().strip("]").strip("[").strip("{").strip("`")
            if param_list_txt == "":
                continue
            else:
                tmp = {}
                for param_txt in param_list_txt.split(","):
                    tmp[param_txt.split(":")[0].strip().strip("{").strip("\"").strip()] = param_txt.split(":")[1].strip().strip("\"").strip()
                if 'alignment score' in tmp:
                    if tmp['alignment score'] in ["N/A", "Not provided"]:
                        tmp['alignment score'] = 0.1
                    elif tmp['alignment score'].lower() == "high":
                        tmp['alignment score'] = 0.8
                    elif tmp['alignment score'].lower() == "moderate":
                        tmp['alignment score'] = 0.5
                    elif tmp['alignment score'].lower() == "low":
                        tmp['alignment score'] = 0.2
                    tmp['alignment score'] = float(tmp['alignment score'])
                if 'center' in tmp:
                    tmp['center'] = bool(tmp['center'])
                if "language" in tmp:
                    if tmp["language"] in APPEND_DICT.keys():
                        if APPEND_DICT[tmp["language"]] not in input_data[lang][i]:
                            print("requesting...")
                            # tmp["language"] = "English"
                            more_data[APPEND_DICT[tmp["language"]]] = clp_request(i, mgsm_data.input_data[lang][i], self.LANG_DICT[lang], tmp["language"])
                            request_flag = True
                            print("finished...")
                        tmp["language"] = APPEND_DICT[tmp["language"]]
                    else:
                        tmp["language"] = self.reverse_dict[tmp["language"]]
                else:
                    print("error")
                temp_list.append(tmp)
        more_data["raw"] = respond_txt
        more_data.update({"params": temp_list})
        return more_data, request_flag
     
    def load_from_merge(self, select_path, pred_path):
        self.load_from_saved("mgsm/output/auto-clsp")
        input_data = self.input_data
        raw_data = {}
        APPEND_DICT = {
            "Hindi": "hi",
            "Portuguese": "por",
            "Dutch": "du",
            "Italian": "it",
            "Korean": "ko",
            "Polish": "pol",
            "Czech": "cz",
            "Vietnamese": "vi",
            "Catalan": "ca",
            "Tamil": "ta",
            "Kannada": "ka",
        }
        mgsm_data = MGSM("mgsm/input")
        request_flag = False
        for lang in tqdm(self.LANG_DICT):
        # for lang in tqdm(["bn", "fr", "zh"]):
            if lang not in input_data:
                input_data[lang] = []
            if lang not in raw_data:
                raw_data[lang] = []
            request_flag = False
            if os.path.exists(os.path.join(select_path, f"request_{lang}_res.jsonl")):
                with open(os.path.join(select_path, f"request_{lang}_res.jsonl"), "r", encoding="utf8") as f:
                        for i, line1 in enumerate(tqdm(f, total=250)):
                            
                            respond_txt = json.loads(line1.strip())[1]["choices"][0]["message"]["content"]
                            idx = json.loads(line1.strip())[-1]["row_id"]
                            more_data, temp_request_flag = self.extract_languages(idx, respond_txt, APPEND_DICT, lang, input_data, mgsm_data)
                            if temp_request_flag:
                                request_flag=True
                            if len(input_data[lang]) > idx:
                                input_data[lang][idx].update(more_data)
                            else:
                                while len(input_data[lang]) <= idx:
                                    input_data[lang].append({})
                                input_data[lang][idx].update(more_data)
                            if len(raw_data[lang]) > idx:
                                raw_data[lang][idx] = json.loads(line1.strip())
                            else:
                                while len(raw_data[lang]) <= idx:
                                    raw_data[lang].append("")
                                raw_data[lang][idx] = json.loads(line1.strip())

            for lang2 in list(self.LANG_DICT.keys())+["en"]:
                with open(os.path.join(pred_path, f"{lang}/{lang2}.jsonl"), "r", encoding="utf8") as f:
                    for i, line in enumerate(f):
                        if lang2 in [x["language"] for x in input_data[lang][i]["params"]]:
                            input_data[lang][i][lang2] = json.loads(line.strip())
            if request_flag:
                self.input_data = input_data
                self.save_data("mgsm/output/auto-clsp")
        self.input_data = input_data
        self.raw_data = raw_data
        if request_flag:
            self.save_data("mgsm/output/auto-clsp")
        self.reverse_dict.update(APPEND_DICT)
        
    def load_from_saved(self, save_dir):
        input_data = {}
        for lang in self.LANG_DICT:
            if os.path.exists(os.path.join(save_dir, f"{lang}.jsonl")):
                input_data[lang] = read_jsonl(os.path.join(save_dir, f"{lang}.jsonl"))
        self.input_data = input_data
    
    def save_data(self, save_dir):
        for lang in self.LANG_DICT:
            print("Saving to " + os.path.join(save_dir, f"{lang}.jsonl"))
            if lang in self.input_data and len(self.input_data) != 0:
                write_jsonl(os.path.join(save_dir, f"{lang}.jsonl"), self.input_data[lang], "w")
    
    def get_raw_data(self, lang, idx):
        return self.raw_data[lang][idx]
    
    def update_raw_data(self, lang, idx, value):
        assert self.raw_data[lang][idx][0]["messages"][0]["content"] == value[0]["messages"][0]["content"]
        self.raw_data[lang][idx][1]["choices"][0]["message"]["content"] = value[1]["choices"][0]["message"]["content"]
        self.raw_data[lang][idx][0]["messages"][1]["content"] = value[0]["messages"][1]["content"]
    
    def save_raw_selection(self, save_dir, lang=None):
        if lang is None:
            for lang in self.LANG_DICT:
                if lang in self.raw_data and len(self.raw_data) != 0 and not os.path.exists(os.path.join(save_dir, f"request_{lang}_res.jsonl")):
                    print("Saving to " + os.path.join(save_dir, f"request_{lang}_res.jsonl"))
                    write_jsonl(os.path.join(save_dir, f"request_{lang}_res.jsonl"), self.raw_data[lang], "w")
        else:
            if lang in self.raw_data and len(self.raw_data) != 0 and not os.path.exists(os.path.join(save_dir, f"request_{lang}_res.jsonl")):
                print("Saving to " + os.path.join(save_dir, f"request_{lang}_res.jsonl"))
                write_jsonl(os.path.join(save_dir, f"request_{lang}_res.jsonl"), self.raw_data[lang], "w")