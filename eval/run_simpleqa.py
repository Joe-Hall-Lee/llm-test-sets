import json
import os
import pandas as pd
from tqdm import tqdm

import common
from simpleqa_eval import SimpleQAEval


def main():
    debug = False
    samplers = {
        # "gemini-1.5": "gemini-1.5",
        # 'yi-lightning': 'yi-lightning',
        # 'glm-4-plus': 'glm-4-plus',
        # "ernie-4.0-turbo": "ernie-4.0-turbo",
        # 'qwen-plus': 'qwen-plus',
        "doubao-pro": "doubao-pro"
    }

    def get_evals():
        return SimpleQAEval(num_examples=10 if debug else None)

    evals = {
        eval_name: get_evals() for eval_name in ["simpleqa"]
    }

    print(evals)
    debug_suffix = "_DEBUG" if debug else ""
    mergekey2resultpath = {}

    for sampler_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            with tqdm(total=1, desc=f"Running {eval_name} with {sampler_name}", unit="eval") as pbar:
                result = eval_obj(sampler)
                pbar.update(1)

            # Write report after evaluation
            file_stem = f"{eval_name}_{sampler_name}"
            report_filename = f"../result/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            os.makedirs(os.path.dirname(report_filename), exist_ok=True)

            with open(report_filename, "w", encoding="utf-8") as fh:
                fh.write(common.make_report(result))

            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"../result/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename

    merge_metrics = []
    for eval_sampler_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_sampler_name[:eval_sampler_name.find("_")]
        sampler_name = eval_sampler_name[eval_sampler_name.find("_") + 1:]
        merge_metrics.append(
            {"eval_name": eval_name, "sampler_name": sampler_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["sampler_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())

    return merge_metrics


if __name__ == "__main__":
    main()
