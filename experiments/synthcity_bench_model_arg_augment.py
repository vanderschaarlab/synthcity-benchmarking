import argparse
import pickle
from pathlib import Path

import pandas as pd
import synthcity.logger as log
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader

log.add("synthcity_logs", "INFO")

KWARGS = {"n_iter": 100}
KWARGS_str = "-".join([f"{k}:{v}" for k, v in KWARGS.items()])


def run_dataset(loader, workspace_path, model):
    try:
        score = Benchmarks.evaluate(
            [(model, model, KWARGS)],
            loader.train(),
            loader.test(),
            task_type="classification",
            synthetic_size=loader.dataframe().shape[0],
            synthetic_reuse_if_exists=False,
            augmented_reuse_if_exists=False,
            augmentation_rule="equal",
            metrics={
                "performance": [
                    "linear_model_augmentation",
                    "mlp_augmentation",
                    "xgb_augmentation",
                ],
            },
            workspace=workspace_path,
            repeats=5,
        )
        print(score)
    except Exception as e:
        print("\n\n", e)
        print(workspace_path, model, "classification", KWARGS)
        score = None

    return score


def run_synthcity(model="ctgan"):
    file_path = f"../data/augmentation/"
    workspace_path = Path("../workspace/augmentation/")
    result_path = f"../results/augmentation/{model}"
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # Load and Prep data
    file = "covid_normalised_numericalised.csv"
    print(f"{file_path}/{file}")
    X = pd.read_csv(f"{file_path}/{file}")
    time_horizon = 14
    X.loc[
        (X["Days_hospital_to_outcome"] <= time_horizon) & (X["is_dead"] == 1),
        f"is_dead_at_time_horizon={time_horizon}",
    ] = 1
    X.loc[
        (X["Days_hospital_to_outcome"] > time_horizon),
        f"is_dead_at_time_horizon={time_horizon}",
    ] = 0
    X.loc[(X["is_dead"] == 0), f"is_dead_at_time_horizon={time_horizon}"] = 0
    X[f"is_dead_at_time_horizon={time_horizon}"] = X[
        f"is_dead_at_time_horizon={time_horizon}"
    ].astype(int)

    X.drop(columns=["is_dead", "Days_hospital_to_outcome"], inplace=True)

    loader = GenericDataLoader(
        X,
        target_column=f"is_dead_at_time_horizon={time_horizon}",
        sensitive_features=["Ethnicity"],
        fairness_column="Ethnicity",
        domain_column="Ethnicity",
        random_state=42,
    )

    score = run_dataset(loader, workspace_path, model)
    if score:
        with open(f"{result_path}/{file}-{model}-{KWARGS_str}.pkl", "wb") as f:
            pickle.dump(score, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="ctgan",
        choices=["ctgan", "ddpm", "tvae", "goggle", "radialgan"],
    )

    args = parser.parse_args()
    run_synthcity(args.model)
