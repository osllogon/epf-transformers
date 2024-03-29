# deep learning imports
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from epftoolbox.data import read_data
from epftoolbox.models import DNN
from epftoolbox.models._dnn import _build_and_split_XYs

# other imports
import random
import os
from typing import Tuple, Dict

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ElectricDataset(Dataset):
    def __init__(
        self, dataset: pd.DataFrame, sequence_length: int, evaluate: bool = False
    ) -> None:
        """
        Constructor of ElectricDataset

        Args:
            dataset: dataset in dataframe format. It has three columns (prie, feature 1, feature 2) and the index is
                    Timedelta format
            sequence_length: number of past hours to extract for each item of the dataset
            evaluate: if true then only evaluate at the beginning of each day (00:00 hour)
        """

        # set object attributes
        self.dataset = torch.from_numpy(dataset.to_numpy()).to(device)
        dates = torch.zeros((len(dataset), 3)).to(device)
        dates[:, 0] = torch.Tensor(dataset.index.hour.to_numpy())
        dates[:, 1] = torch.Tensor(dataset.index.dayofweek.to_numpy())
        dates[:, 2] = torch.Tensor(dataset.index.month.to_numpy())
        self.dataset = torch.concat((self.dataset, dates), dim=1)
        self.sequence_length = sequence_length
        self.evaluate = evaluate

    def __len__(self) -> int:
        """
        This method returns the length of the dataset

        Returns:
            number of valid items of the dataset
        """
        if self.evaluate:
            return (self.dataset.size(0) - self.sequence_length) // 24
        else:
            return self.dataset.size(0) - self.sequence_length - 23

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        This method returns an element from the dataset based on the index

        Args:
            index: index of the element

        Returns:
            values of the last sequence length hours and the next 24 hours than the index
        """

        # extract values
        if self.evaluate:
            values = self.dataset[24 * index : 24 * index + self.sequence_length + 24]
        else:
            values = self.dataset[index : index + self.sequence_length + 24]

        return values


def load_data(
    dataset_option: str,
    save_path: str,
    sequence_length: int = 72,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    This method returns Dataloaders of the chosen dataset.

    Args:
        dataset_option: path for the dataset.
        save_path: path to save the data.
        sequence_length : sequence length for the neural net, It has to be a multiple of 24.
        batch_size: size of batches that wil be created.
        shuffle: indicator of shuffle the data.
        drop_last: indicator to drop the last batch since it is not full.
        num_workers: num workers for loading the data.

    Returns:
        train dataloader
        val dataloader
        test dataloader
        means of price
        stds of price
    """

    # create dir if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # download dataset
    url_dir: str = "https://zenodo.org/records/4624805/files/"
    data: pd.DataFrame = pd.read_csv(url_dir + dataset_option + ".csv", index_col=0)
    file_path: str = os.path.join(save_path, dataset_option + ".csv")
    data.to_csv(file_path)

    # create initial datasets
    df_train_val: pd.DataFrame
    df_test: pd.DataFrame
    df_train_val, df_test = read_data(
        dataset=dataset_option, path=save_path, years_test=2
    )
    df_test = pd.concat([df_train_val[-sequence_length:], df_test], axis=0)
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for variable in df_train_val.columns:
        means[variable] = df_train_val[variable].mean()
        stds[variable] = df_train_val[variable].std()
        df_train_val[variable] = (df_train_val[variable] - means[variable]) / stds[
            variable
        ]

    df_train: pd.DataFrame = df_train_val[: -42 * 7 * 24]
    df_val: pd.DataFrame = df_train_val[-42 * 7 * 24 :]
    df_val = pd.concat([df_train[-sequence_length:], df_val], axis=0)
    train_dataset: Dataset = ElectricDataset(df_train, sequence_length)
    val_dataset: Dataset = ElectricDataset(df_val, sequence_length, evaluate=True)
    test_dataset: Dataset = ElectricDataset(df_test, sequence_length, evaluate=True)

    # define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        means["Price"],
        stds["Price"],
    )


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: seed for deterministic behavior
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def forecast_next_day(model: DNN, df: pd.DataFrame, next_day_date) -> np.ndarray:
    """
    Method that builds an easy-to-use interface for daily recalibration and forecasting of the DNN model

    The method receives a pandas dataframe ``df`` and a day ``next_day_date``. Then, it
    recalibrates the model using data up to the day before ``next_day_date`` and makes a prediction
    for day ``next_day_date``.

    Args:
        df: Dataframe of historical data containing prices and N exogenous inputs. The index of the
            dataframe should be dates with hourly frequency. The columns should have the following
            names ['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']
        next_day_date : TYPE
            Date of the day-ahead

    Returns:
        An array containing the predictions in the provided date

    """

    # We define the new training dataset considering the last calibration_window years of data
    df_train = df.loc[: next_day_date - pd.Timedelta(hours=1)]
    df_train = df_train.loc[
        next_day_date - pd.Timedelta(hours=model.calibration_window * 364 * 24) :
    ]

    # We define the test dataset as the next day (they day of interest) plus the last two weeks
    # in order to be able to build the necessary input features.
    df_test = df.loc[next_day_date - pd.Timedelta(weeks=2) :, :]

    # Generating training, validation, and test input and outpus. For the test dataset,
    # even though the dataframe contains 15 days of data (next day + last 2 weeks),
    # we provide as parameter the date of interest so that Xtest and Ytest only reflect that
    Xtrain, Ytrain, Xval, Yval, Xtest, _, _ = _build_and_split_XYs(
        dfTrain=df_train,
        features=model.best_hyperparameters,
        shuffle_train=True,
        dfTest=df_test,
        date_test=next_day_date,
        data_augmentation=model.data_augmentation,
        n_exogenous_inputs=len(df_train.columns) - 1,
    )

    # Normalizing the input and outputs if needed
    Xtrain, Xval, Xtest, Ytrain, Yval = model._regularize_data(
        Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain, Yval=Yval
    )

    # Recalibrating the neural network and extracting the prediction
    Yp = model.predict(Xtest=Xtest)

    return Yp
