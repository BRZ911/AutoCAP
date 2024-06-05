from utils.datasets import MGSM

INPUT_DIR = "mgsm/input"
OUTPUT_DIR = "auto-clsp-exp/mgsm/l6-01-tp02-tp02"
TEMPERATURE_1 = 0.2
TEMPERATURE_2 = 0.2
dataset = MGSM(INPUT_DIR)
dataset.parse_language_choice_request_stage_1(OUTPUT_DIR, temperature=TEMPERATURE_1)
dataset.request_loop(step=0)
dataset.parse_language_choice_request_stage_2(f"{OUTPUT_DIR}/output",
                                              OUTPUT_DIR,
                                              temperature=TEMPERATURE_2)
dataset.request_loop(step=1)