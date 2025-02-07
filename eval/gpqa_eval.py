import re

import os
import pandas as pd

import common
from common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from type import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult
from call_model import call_model


class GPQAEval:
    def __init__(
            self,
            n_repeats: int = 1,
            variant: str = "diamond",
            num_examples: int | None = None,  # restrict to a subset of the data for debugging
    ):
        data_dir = "/eval/data"
        filename = f"gpqa_{variant}.csv"

        local_path = os.path.join(data_dir, filename)
        df = pd.read_csv(local_path, encoding='latin1')
        self.original_data = df.copy()  # Save original data
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = examples[:num_examples]
        examples = examples * n_repeats
        self.examples = examples
        self.n_repeats = n_repeats
        self.results = {}  # Initialize results as a dictionary

    def __call__(self, sampler):
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )
            prompt_messages = [
                dict(content=format_multichoice_question(choices_dict), role="user"),
            ]
            response_text = call_model(format_multichoice_question(choices_dict), sampler)
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            result_data = {
                "Question": row["Question"],
                "Correct Answer": row["Correct Answer"],
                "Extracted Answer": extracted_answer,
                "Score": score
            }

            # Use question as key to ensure each question has only one set of results
            if row["Question"] not in self.results:
                self.results[row["Question"]] = []
            self.results[row["Question"]].append(result_data)

            return SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)

    def get_results_df(self):
        # Convert dictionary to DataFrame
        all_results = []
        for question, results in self.results.items():
            for result in results:
                all_results.append(result)
        return pd.DataFrame(all_results)