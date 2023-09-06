# deep learning libraries
import torch
import numpy as np
import pandas as pd
from epftoolbox.data import read_data
from epftoolbox.models import DNN
from epftoolbox.models._dnn import _build_and_split_XYs
from epftoolbox.evaluation import MAE, RMSE, sMAPE, DM

# other libraries
import os
from typing import Tuple, List, Dict, Literal

# own libraries
from src.utils import load_data, forecast_next_day

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# static variables
DATASETS_PATH: str = "data"
DATASETS_NAMES: Tuple[str, ...] = ("NP", "PJM", "BE", "FR", "DE")


def main() -> None:
    """
    This function is the train module of the benchmark module

    Raises:
        ValueError: Invalid value for benchmark
    """

    # variables
    benchmark: Literal[
        "dnn_last_year", "dnn_all_past", "naive", "final_results"
    ] = "final_results"

    # compute benchmark
    if benchmark == "dnn_last_year":
        benchmark_dnn_without_retrain(DATASETS_NAMES, True)
    elif benchmark == "dnn_all_past":
        benchmark_dnn_without_retrain(DATASETS_NAMES, False)
    elif benchmark == "naive":
        benchmark_naive_daily_model(DATASETS_NAMES)
    elif benchmark == "final_results":
        final_results(DATASETS_NAMES)
    else:
        raise ValueError("Invalid value for benchmark")


def benchmark_dnn_without_retrain(
    datasets: Tuple[str, ...], last_year: bool = False
) -> None:
    """
    Benchmark for dnn without retraining

    Args:
        datasets: name of the datasets to use.
        last_year: bool to indicate if renormalized is over the last year data or all the past data. Defaults to False.
    """

    # define metrics vector
    results: List[List[float]] = []

    # iterate over datasets
    dataset: str
    for dataset in datasets:
        df_train: pd.DataFrame
        df_test: pd.DataFrame
        df_train, df_test = read_data(
            f"data/{dataset}",
            dataset=dataset,
            years_test=2,
            begin_test_date=None,
            end_test_date=None,
        )

        # define model
        model: DNN = DNN(
            experiment_id="1",
            path_hyperparameter_folder="epftoolbox/examples/experimental_files",
            nlayers=2,
            dataset=dataset,
            years_test=2,
            shuffle_train=True,
            data_augmentation=0,
            calibration_window=4,
        )

        # compute real values
        forecast_dates = df_test.index[::24]
        forecast = pd.DataFrame(
            index=df_test.index[::24], columns=["h" + str(k) for k in range(24)]
        )
        real_values = df_test.loc[:, ["Price"]].values.reshape(-1, 24)
        real_values = pd.DataFrame(
            real_values, index=forecast.index, columns=forecast.columns
        )

        # start to calibrate model with train data
        Xtrain, Ytrain, Xval, Yval, Xtest, _, _ = _build_and_split_XYs(
            dfTrain=df_train,
            features=model.best_hyperparameters,
            shuffle_train=True,
            dfTest=df_test,
            date_test=None,
            data_augmentation=model.data_augmentation,
            n_exogenous_inputs=len(df_train.columns) - 1,
        )

        # Normalizing the input and outputs if needed
        Xtrain, Xval, Xtest, Ytrain, Yval = model._regularize_data(
            Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain, Yval=Yval
        )

        # renormalize
        model.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval)

        # for loop over the recalibration dates
        for date in forecast_dates:
            # For simulation purposes, we assume that the available data is
            # the data up to current date where the prices of current date are not known
            if last_year:
                data_available = pd.concat(
                    [
                        df_train[-365 * 24 :],
                        df_test.loc[: date + pd.Timedelta(hours=23), :],
                    ],
                    axis=0,
                )
            else:
                data_available = pd.concat(
                    [df_train, df_test.loc[: date + pd.Timedelta(hours=23), :]], axis=0
                )

            # We extract real prices for current date and set them to NaN in the dataframe of available data
            data_available.loc[date : date + pd.Timedelta(hours=23), "Price"] = np.NaN

            # forecast for next day
            Yp: np.ndarray = forecast_next_day(
                model, df=data_available, next_day_date=date
            )

            # Saving the current prediction
            forecast.loc[date, :] = Yp

        # computing metrics up-to-current-date
        mae: float = MAE(real_values.values, forecast.values)
        rmse: float = RMSE(real_values.values, forecast.values)
        smape: float = sMAPE(real_values.values, forecast.values)

        # appending results
        results.append([mae, rmse, smape])

        # saving forecast
        if not os.path.isdir(
            f"results/forecasts/dnn_ensemble_without_retraining_"
            f"last_year_{last_year}"
        ):
            os.makedirs(
                f"results/forecasts/dnn_ensemble_without_retraining_last_year_{last_year}"
            )
        forecast.to_csv(
            f"results/forecasts/dnn_ensemble_without_retraining_last_year_{last_year}/"
            f"{dataset}.csv"
        )

    # save results
    df: pd.DataFrame = pd.DataFrame(
        data=np.array(results), index=datasets, columns=["MAE", "RMSE", "sMAPE"]
    )
    if not os.path.isdir("results/benchmarks"):
        os.makedirs("results/benchmarks")
    df.to_csv(
        f"results/benchmarks/dnn_ensemble_without_retraining_last_year_{last_year}.csv"
    )

    return None


@torch.no_grad()
def benchmark_naive_daily_model(datasets: Tuple[str, ...]) -> None:
    """
    This function computes a benchmark predicting the next day as the present day values

    Args:
        datasets: names of the datasets to use
    """

    # define metrics vector
    results = []

    # iterate over datasets
    dataset: str
    for dataset in datasets:
        # load test data
        df_train: pd.DataFrame
        df_test: pd.DataFrame
        df_train, df_test = read_data(
            f"{DATASETS_PATH}/{dataset}",
            dataset=dataset,
            years_test=2,
            begin_test_date=None,
            end_test_date=None,
        )
        df_test = pd.concat([df_train[-24:], df_test], axis=0)

        # define teste number of days
        test_number_of_days: int = len(df_test) // 24

        # define forecast object
        real_values: np.ndarray = np.zeros((test_number_of_days - 1, 24))
        forecast: np.ndarray = np.zeros((test_number_of_days - 1, 24))

        # iter over test days
        for i in range(test_number_of_days - 1):
            real_values[i] = df_test["Price"][24 + 24 * i : 48 + 24 * i].to_numpy()
            forecast[i] = df_test["Price"][24 * i : 24 * i + 24].to_numpy()

        # computing metrics up-to-current-date
        mae: float = MAE(real_values, forecast)
        rmse: float = RMSE(real_values, forecast)
        smape: float = sMAPE(real_values, forecast)

        # add dataset results
        results.append([mae, rmse, smape])

        # saving forecast
        if not os.path.isdir(f"results/forecasts/dump_daily_model"):
            os.makedirs(f"results/forecasts/naive_daily_model")
        df_forecast = pd.DataFrame(
            index=df_test.index[24::24],
            columns=["h" + str(k) for k in range(24)],
            data=forecast,
        )
        df_forecast.to_csv(f"results/forecasts/naive_daily_model/{dataset}.csv")

    df: pd.DataFrame = pd.DataFrame(
        data=np.round_(np.array(results), decimals=3),
        index=datasets,
        columns=["MAE", "RMSE", "sMAPE"],
    )
    if not os.path.isdir("results/benchmarks"):
        os.makedirs("results/benchmarks")
    df.to_csv(f"results/benchmarks/naive_daily_model.csv")

    return None


@torch.no_grad()
def final_results(datasets: Tuple[str, ...]) -> None:
    """
    This function computes the final table of results

    Args:
        datasets: names of the datasets to use
    """

    results = []

    for dataset in datasets:
        # load data
        train_data, val_data, test_data, mean, std = load_data(
            dataset, f"{DATASETS_PATH}/{dataset}", sequence_length=128, batch_size=128
        )

        directories = os.listdir(f"best_models/{dataset}")
        model_path = (
            directories[1] if directories[0].split(".")[-1] == "csv" else directories[0]
        )
        # model_type = directories[0].split('_')[0]
        model = torch.jit.load(f"best_models/{dataset}/{model_path}")

        # evaluate
        model.eval()

        # initialize losses vectors
        real_values = torch.zeros((test_data.__len__(), 24))

        # val loop
        i = 0
        for inputs in test_data:
            # prepare data
            inputs = inputs.float()
            targets = inputs[:, -24:, 0]

            real_values[i] = targets[0, -24:].detach().cpu()
            i += 1

        # compute measures
        real_values = real_values.numpy()

        forecast = pd.read_csv(f"best_models/{dataset}/forecast.csv")
        forecast = forecast.to_numpy()[:, 1:]

        mae = MAE(real_values, forecast)
        rmse = RMSE(real_values, forecast)
        smape = sMAPE(real_values, forecast)

        forecast_dnn_ensemble = pd.read_csv(
            f"results/forecasts/dnn_ensemble_without_retraining_last_"
            f"year_False/{dataset}.csv"
        )
        forecast_dnn_ensemble = forecast_dnn_ensemble.to_numpy()[:, 1:]
        dm_test_dnn_ensemble = DM(
            real_values, forecast_dnn_ensemble, forecast, version="multivariate"
        )
        forecast_dnn_ensemble_last_year = pd.read_csv(
            f"results/forecasts/dnn_ensemble_without_"
            f"retraining_last_year_True/{dataset}.csv"
        )
        forecast_dnn_ensemble_last_year = forecast_dnn_ensemble_last_year.to_numpy()[
            :, 1:
        ]
        dm_test_dnn_ensemble_last_year = DM(
            real_values,
            forecast_dnn_ensemble_last_year,
            forecast,
            version="multivariate",
        )
        forecast_naive_model = pd.read_csv(
            f"results/forecasts/naive_daily_model/{dataset}.csv"
        )
        forecast_naive_model = forecast_naive_model.to_numpy()[:, 1:]
        dm_test_naive_model = DM(
            real_values, forecast_naive_model, forecast, version="multivariate"
        )

        results.append(
            [
                mae,
                rmse,
                smape,
                dm_test_dnn_ensemble,
                dm_test_dnn_ensemble_last_year,
                dm_test_naive_model,
            ]
        )

    df = pd.DataFrame(
        data=np.round_(np.array(results), decimals=3),
        index=datasets,
        columns=[
            "MAE",
            "RMSE",
            "sMAPE",
            "DM test DNN",
            "DM test DNN last year",
            "DM test Dum Model",
        ],
    )
    if not os.path.isdir(f"results"):
        os.makedirs(f"results")
    df.to_csv(f"results/results.csv")

    return None


if __name__ == "__main__":
    main()
