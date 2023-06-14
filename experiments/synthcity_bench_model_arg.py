import argparse
import os
import pickle
from pathlib import Path

import synthcity.logger as log
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader

log.add("synthcity_logs", "INFO")

KWARGS = {"n_iter": 100}
KWARGS_str = "-".join([f"{k}:{v}" for k, v in KWARGS.items()])


def run_dataset(X, workspace_path, model, task_type="regression"):
    loader = GenericDataLoader(X, target_column="y")

    try:
        score = Benchmarks.evaluate(
            [(model, model, KWARGS)],
            loader.train(),
            loader.test(),
            task_type=task_type,
            synthetic_size=X.shape[0],
            metrics={
                "stats": ["alpha_precision"],
                "detection": ["detection_xgb", "detection_mlp", "detection_linear"],
                "performance": ["linear_model", "mlp", "xgb"],
            },
            workspace=workspace_path,
            repeats=1,
            device="cpu",
        )
    except Exception as e:
        print("\n\nSkipping dataset: ", e)
        score = None

    return score


def run_synthcity(data_type="num", task_type="regression", model="ctgan"):
    cwd = Path.cwd()
    if cwd.name == "experiments":
        cwd = cwd.parent
    file_path = (cwd / Path(f"data/{data_type}/{task_type}/")).resolve()
    workspace_path = (cwd / Path("workspace/{data_type}/{task_type}/")).resolve()
    result_path = (cwd / Path(f"results/{data_type}/{task_type}/{model}")).resolve()
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # list files in the file_path
    files = os.listdir(file_path)
    print(f"Number of files in {file_path}: {len(files)}")

    for file in files:
        print(f"{file_path}/{file}")
        with open(f"{file_path}/{file}", "rb") as f:
            data_dict = pickle.load(f)

        X = data_dict["X"]
        y = data_dict["y"]
        X["y"] = y

        score = run_dataset(X, workspace_path, model, task_type=task_type)
        print(f"{result_path}/{file}-{model}-{KWARGS_str}.pkl")
        if score:
            print(score)
            with open(f"{result_path}/{file}-{model}-{KWARGS_str}.pkl", "wb") as f:
                pickle.dump(score, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="num", choices=["num", "cat"])
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ctgan",
        choices=["ctgan", "ddpm", "tvae", "goggle", "arf"],
    )

    args = parser.parse_args()
    run_synthcity(args.data_type, args.task_type, args.model)
