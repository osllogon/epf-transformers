# deep learning libraries
import torch
import numpy as np
import pandas as pd

# other libraries
import random
import os
import matplotlib.pyplot as plt
from typing import Tuple

# own modules
from src.utils import set_seed

# set seed
set_seed(42)

# static variables
FORECASTS_PATH: str = "results/forecasts"
DATASETS_NAMES: Tuple[str, ...] = ("NP", "PJM", "BE", "FR", "DE")


def main() -> None:
    """
    Main function to create visualizations
    """

    # create visualizations
    draw_forecasts(DATASETS_NAMES, "visualizations")


def draw_forecasts(
    datasets: Tuple[str, ...], save_path: str, number_per_day: int = 2
) -> None:
    """
    This function creates visualizations of the forecasts of the models

    Args:
        datasets: list of datasets
        number_per_day: number of visualizations per day of the week
    """

    for dataset in datasets:
        # load results
        real_values_df: pd.DataFrame = pd.read_csv(
            f"{FORECASTS_PATH}/naive_daily_model/{dataset}.csv"
        )
        dnn_last_year_df: pd.DataFrame = pd.read_csv(
            f"{FORECASTS_PATH}/dnn_ensemble_without_retraining_last_year_True/{dataset}.csv"
        )
        dnn_all_years_df: pd.DataFrame = pd.read_csv(
            f"{FORECASTS_PATH}/dnn_ensemble_without_retraining_last_year_False/{dataset}.csv"
        )
        transformers_df: pd.DataFrame = pd.read_csv(
            f"best_models/{dataset}/forecast.csv"
        )

        # get random index
        index = random.randint(1, dnn_last_year_df.shape[1])

        # get results
        real_values: np.ndarray = real_values_df.to_numpy()[index + 1, 1:]
        dnn_last_year_results: np.ndarray = dnn_last_year_df.to_numpy()[index, 1:]
        dnn_all_years_results: np.ndarray = dnn_all_years_df.to_numpy()[index, 1:]
        transformers_results: np.ndarray = transformers_df.to_numpy()[index, 1:]

        # create figure
        plt.figure()
        plt.plot(np.arange(24), real_values)
        plt.plot(np.arange(24), transformers_results)
        plt.plot(np.arange(24), dnn_last_year_results)
        plt.plot(np.arange(24), dnn_all_years_results)
        plt.xlabel("time [hour]")
        plt.ylabel("Price [EUR/MWh]")
        plt.legend(["Real values", "Transformer", "DNN Ensemble 1", "DNN Ensemble 2"])

        # save and close figure
        if not os.path.isdir(f"{save_path}"):
            os.makedirs(f"{save_path}")
        plt.savefig(f"{save_path}/{dataset}.pdf")
        plt.close()

    return None


if __name__ == "__main__":
    main()
