import argparse
import pickle
from pathlib import Path

import synthcity.logger as log
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.utils.datasets.time_series.google_stocks import \
    GoogleStocksDataloader
from synthcity.utils.datasets.time_series.pbc import PBCDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader

log.add("synthcity_logs", "INFO")

KWARGS = {"n_iter": 100}
KWARGS_str = "-".join([f"{k}:{v}" for k, v in KWARGS.items()])


def run_dataset(loader, workspace_path, model):
    try:
        score = Benchmarks.evaluate(
            [(model, model, KWARGS)],
            loader.train(),
            loader.test(),
            task_type="time_series",
            synthetic_size=loader.dataframe().shape[0],
            synthetic_reuse_if_exists=False,
            augmented_reuse_if_exists=False,
            metrics={
                "stats": ["alpha_precision"],
                "detection": ["detection_xgb", "detection_mlp", "detection_linear"],
                "performance": [
                    "linear_model",
                    "mlp",
                    "xgb",
                ],
            },
            workspace=workspace_path,
            repeats=2,
        )
        print(score)
    except Exception as e:
        print("\n\n", e)
        print(workspace_path, model, "time_series", KWARGS)
        score = None

    return score


def run_synthcity(model="ctgan", dataset_loader_name="sine"):
    dataset_loaders = {
        "sine": SineDataloader,
        "googlestocks": GoogleStocksDataloader,
        "pbc": PBCDataloader,
    }
    file_path = f"../data/time_series/"
    workspace_path = Path("../workspace/time_series/")
    result_path = f"../results/time_series/{model}"
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # Load and Prep data
    (
        static_data,
        temporal_data,
        observation_times,
        outcome,
    ) = dataset_loaders[dataset_loader_name.lower()]().load()
    if dataset_loader_name.lower() == "pbc":
        T, E = outcome
        loader = TimeSeriesDataLoader(
            temporal_data=temporal_data,
            observation_times=observation_times,
            static_data=static_data,
            T=T,
            E=E,
        )
    else:
        loader = TimeSeriesDataLoader(
            temporal_data=temporal_data,
            observation_times=observation_times,
            static_data=static_data,
            outcome=outcome,
        )
    print(loader.dataframe().head())

    score = run_dataset(loader, workspace_path, model)
    if score:
        with open(
            f"{result_path}/{dataset_loader_name.lower()}-{model}-{KWARGS_str}.pkl",
            "wb",
        ) as f:
            pickle.dump(score, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="timegan",
        choices=["timegan", "fflows", "timevae"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="googlestocks",
        choices=["googlestocks", "sine", "pbc"],
    )

    args = parser.parse_args()
    run_synthcity(args.model, args.dataset)
