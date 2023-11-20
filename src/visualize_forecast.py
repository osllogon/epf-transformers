# deep learning libraries
import torch
import numpy as np
import pandas as pd
from epftoolbox.data import read_data

# other libraries
import os
import datetime
from typing import List
from random import randint
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# own modules
from src.utils import load_data, set_seed

# static variables
DATASETS_PATH = "data"
RESULTS_PATH: str = "results"

# set seed
set_seed(42)


def draw_forecasts(
    datasets: List[str], save_path: str, number_per_day: int = 2
) -> None:
    """
    This function creates visualizations of the forecasts of the models

    Args:
        datasets: list of datasets
        number_per_day: number of visualizations per day of the week
    """

    for dataset in datasets:
        # load forecasts
        model_forecast: np.ndarray = pd.read_csv(
            f"best_models/{dataset}/forecast.csv"
        ).to_numpy()
        naive_forecast: np.ndarray = pd.read_csv(
            f"{RESULTS_PATH}/forecasts/naive_daily_model/{dataset}.csv"
        ).to_numpy()
        dnn_retrain_all_years: np.ndarray = pd.read_csv(
            f"{RESULTS_PATH}/forecasts/dnn_ensemble_without_retraining_last_year_False/{dataset}.csv"
        ).to_numpy()
        dnn_retrain_last_year: np.ndarray = pd.read_csv(
            f"{RESULTS_PATH}/forecasts/dnn_ensemble_without_retraining_last_year_True/{dataset}.csv"
        ).to_numpy()

        # get index
        index: int = randint(0, len(model_forecast) - 1)

        # get init time
        start: datetime.datetime = datetime.datetime.strptime(
            naive_forecast[index + 1, 0], "%Y-%m-%d"
        )
        times_array: np.ndarray = np.array(
            [start + datetime.timedelta(hours=i) for i in range(24)]
        )

        # draw visualization
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(times_array, naive_forecast[index + 1, 1:])
        plt.plot(times_array, model_forecast[index, 1:])
        plt.plot(times_array, naive_forecast[index, 1:])
        plt.plot(times_array, dnn_retrain_all_years[index, 1:])
        plt.plot(times_array, dnn_retrain_last_year[index, 1:])
        plt.xlabel("time [hour]")
        ax.xaxis.set_major_formatter(DateFormatter("%H:00"))
        plt.gcf().autofmt_xdate()
        plt.ylabel("Price [EUR/MWh]")
        plt.legend(
            [
                "Real values",
                "Transformer",
                "Naive model",
                "DNN retrained with all years",
                "DNN retariend with last year",
            ]
        )
        plt.grid()

        # save and close figure
        if not os.path.isdir(f"{save_path}"):
            os.makedirs(f"{save_path}")
        plt.savefig(f"{save_path}/{dataset}.pdf")
        plt.close()

    return None


if __name__ == "__main__":
    draw_forecasts(["NP", "PJM", "BE", "FR", "DE"], "visualizations")
