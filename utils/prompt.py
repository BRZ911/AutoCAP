class LanguageChoicePrompt():
    def __init__(self) -> None:
        self.prompt_template = \
"""As an expert in multi-lingual understanding, your task is to select at least three languages optimal for cross-lingual reasoning on a given {} sample. Consider language family, branch, and pre-training data proportions.

**Step-by-Step Instructions:**

1. **Selection Rationale**: Briefly explain why you chose each language based on their linguistic characteristics and pre-training data proportions.
2. **Alignment Score Calculation**: For each selected language, assign an alignment score (ranging from 0 to 1). This score should reflect the language's compatibility with others based on family, branch, and data proportion.
3. **Centric Language Identification**: Based on your analysis, designate one language as the central or pivot language.
4. **Conclusion**: Summarize your selections using the format:
   `Target Language=[{{"language": "L1", "alignment score": "S1", "center": True/False}}, ...]`.

**Language Options:**

- English: Indo-European, Germanic; 46.2%
- Russian: Indo-European, Slavic; 5.8%
- German: Indo-European, Germanic; 5.8%
- French: Indo-European, Romance; 4.7%
- Chinese: Sino-Tibetan, Sinitic; 4.6%
- Spanish: Indo-European, Romance; 4.5%
- Japanese: Japonic, Japanese; 4.4%
- Italian: Indo-European, Romance; 2.7%
- Dutch: Indo-European, Germanic; 2.1%
- Polish: Indo-European, Slavic; 1.6%
- Portuguese: Indo-European, Romance; 1.1%
- Czech: Indo-European, Slavic; 1.1%
- Vietnamese: Austroasiatic, Vietic; 1.1%

**Sample:**
{}
"""
    def generate_prompt(self, lang, input_text):
        return self.prompt_template.format(lang, input_text)

class TwoStageLanguageChoicePrompt():
    def __init__(self) -> None:
        self.prompt_template_1 = \
"""As an expert in multi-lingual understanding, your task is to select six languages optimal for cross-lingual reasoning on a given {} sample. Consider language family, branch, and pre-training data proportions.

**Step-by-Step Instructions:**

1. **Selection Rationale**: Briefly explain why you chose each language based on their linguistic characteristics and pre-training data proportions.
2. **Conclusion**: Summarize your selections using the format:
   `Target Language=[{{"language": "L1"}}, ...]`.

**Language Options:**
- English: Indo-European; Germanic; 46.2%
- Russian: Indo-European; Slavic; 5.8%
- German: Indo-European; Germanic; 5.8%
- French: Indo-European; Romance; 4.7%
- Chinese: Sino-Tibetan; Sinitic; 4.6%
- Spanish: Indo-European; Romance; 4.5%
- Japanese: Japonic; Japanese; 4.4%
- Italian: Indo-European; Romance; 2.7%
- Dutch: Indo-European; Germanic; 2.1%
- Polish: Indo-European; Slavic; 1.6%
- Portuguese: Indo-European; Romance; 1.1%
- Czech: Indo-European; Slavic; 1.1%
- Vietnamese: Austroasiatic; Vietic; 1.1%

**Item Explanation:**
**Language Options** contains language families, language branches and the approximate proportion of pre-trained languages.
Reasoning with the same language family or language branch is beneficial to cross-language alignment and understanding.
Reasoning with the different language family or language branch can have more diverse reasoning paths to obtain better results.
Reasoning with the different proportion of pre-trained languages affects the whole reasoning performance of the model.

**Sample:**
{}"""
        self.prompt_template_2 = \
"""After your language selection, please assign an alignment score for multilingual reasoning aggregation.

**Step-by-Step Instructions:**
1. **Alignment Score Calculation**: For each selected language, assign an alignment score (ranging from 0 to 1). This score should reflect the language's compatibility with others based on family, branch, and data proportion. It is required that the degree of distinction between scores cannot be too small.
2. **Conclusion**: Summarize your selections using the format:
   `Target Language=[{{"language": "L1", "alignment score": a1}}, ...]`.

**Sample:**
{}
"""
    def generate_prompt(self, lang, input_text, step=0):
        if step == 0:
            return [self.prompt_template_1, self.prompt_template_2][step].format(lang, input_text)
        else:
            return [self.prompt_template_1, self.prompt_template_2][step].format(input_text)