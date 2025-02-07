import json
import os
import pandas as pd
from tqdm import tqdm

import common
from gpqa_eval import GPQAEval

def main():
    debug = False
    samplers = {
        # "gemini-1.5": "gemini-1.5",
        # "360gpt2-pro": "360gpt2-pro"
        # "ernie-4.0-turbo": "ernie-4.0-turbo"
        # "doubao-pro": "doubao-pro"
        # "qwen-plus": "qwen-plus"
        # "moonshot-v1": "moonshot-v1"
        "spark4.0-ultra": "spark4.0-ultra"
        # "glm-4-plus": "glm-4-plus",
        # 'yi-lightning': "yi-lightning",
        # "gy-pangu": "gy-pangu",
        # "hunyuan-turbo": "hunyuan-turbo",

    }

    def get_evals():
        return GPQAEval(num_examples=10 if debug else None)

    evals = {
        eval_name: get_evals() for eval_name in ["gpqa"]
    }

    print(evals)
    debug_suffix = "_DEBUG" if debug else ""
    mergekey2resultpath = {}

    # Dictionary to store all evaluation results DataFrame
    all_results = {}
    for sampler_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            with tqdm(total=1, desc=f"Running {eval_name} with {sampler_name}", unit="eval") as pbar:
                result = eval_obj(sampler)
                pbar.update(1)

            # Write report after evaluation
            file_stem = f"{eval_name}_{sampler_name}"
            report_filename = f"F:/CESI/ChallengeBench/result/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            os.makedirs(os.path.dirname(report_filename), exist_ok=True)

            with open(report_filename, "w", encoding="utf-8") as fh:
                fh.write(common.make_report(result))

            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"F:/CESI/ChallengeBench/result/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename

            # 获取当前评估对象的结果 DataFrame 并存储
            if eval_name not in all_results:
                all_results[eval_name] = eval_obj.original_data.copy()
            results_df = eval_obj.get_results_df()
            results_df.set_index('Question', inplace=True)

            # 使用后缀来避免列名冲突
            all_results[eval_name] = pd.merge(
                all_results[eval_name],
                results_df,
                on='Question',
                suffixes=('', '_eval')  # 添加后缀来区分冲突的列
            )

            # 将所有结果写入 CSV 文件
            for eval_name, results_df in all_results.items():
                csv_filename = f"F:/CESI/ChallengeBench/result/{file_stem}{debug_suffix}.csv"
                os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
                results_df.to_csv(csv_filename, index=False, encoding="utf-8")
                print(f"Writing CSV to {csv_filename}")

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