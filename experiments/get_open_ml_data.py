import argparse
import os
import pickle

from openml import tasks
from utils import get_data_id


def run_open_ml(data_type="num", task_type="regression"):

    if task_type == "regression":
        oml_task_type = tasks.TaskType(2)
    else:
        oml_task_type = tasks.TaskType(1)

    file_path = f"../data/{data_type}/{task_type}/"

    # check file_path exists, otherwise create it.
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    task_df = tasks.list_tasks(task_type=oml_task_type, output_format="dataframe")[
        ["tid", "did"]
    ]
    data_id = get_data_id(data_type, task_type)

    task_df_filtered = task_df[task_df.did.isin(data_id)]

    # group by did and take first tid
    task_df_filtered = task_df_filtered.groupby("did").first().reset_index()

    # for each row in task_df_filtered, get the task and the data
    for index, row in task_df_filtered.iterrows():
        tid = row["tid"]
        did = row["did"]
        task = tasks.get_task(tid)
        X, y = task.get_X_and_y(dataset_format="dataframe")
        data_dict = {"X": X, "y": y, "tid": tid, "did": did}

        print(data_dict)

        with open(f"../data/{data_type}/{task_type}/{did}.pkl", "wb") as f:
            pickle.dump(data_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="num", choices=["num", "cat"])
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
    )

    args = parser.parse_args()
    run_open_ml(args.data_type, args.task_type)
