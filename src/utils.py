# deep learning imports
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from epftoolbox.data import read_data
from epftoolbox.models import DNN
from epftoolbox.models._dnn import _build_and_split_XYs
from epftoolbox.evaluation import MAE, RMSE, sMAPE, DM

# other imports
import random
import os
import matplotlib.pyplot as plt

# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# static variables
DATASETS_PATH = './data'


def main() -> None:
    # define datasets
    datasets_names = ['NP', 'PJM', 'BE', 'FR', 'DE']

    # compute results
    benchmark_dnn_without_retrain(datasets_names, False)
    benchmark_naive_daily_model(datasets_names)
    final_results(datasets_names)
    # draw_forecasts(datasets_names, 'visualizations')


class ElectricDataset(Dataset):

    def __init__(self, dataset: pd.DataFrame, sequence_length: int, evaluate: bool = False) -> None:
        """
        Constructor of ElectricDataset

        Parameters
        ----------
        dataset : pd.DataFrame
            dataset in dataframe format. It has three columns (prie, feature 1, feature 2) and the index is Timedelta
            format
        sequence_length : int
            number of past hours to extract for each item of the dataset
        evaluate : bool, Default
            if true then only evaluate at the beginning of each day (00:00 hour)

        Returns
        -------
        None
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

        Returns
        -------
        int
            number of valid items of the dataset
        """
        if self.evaluate:
            return (self.dataset.size(0) - self.sequence_length) // 24
        else:
            return self.dataset.size(0) - self.sequence_length - 23

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        This method returns an element from the dataset based on the index

        Parameters
        ----------
        index : int
            index of the element

        Returns
        -------
        torch.Tensor
            values of the last sequence length hours and the next 24 hours than the index
        """

        # extract values
        if self.evaluate:
            values = self.dataset[24 * index: 24 * index + self.sequence_length + 24]
        else:
            values = self.dataset[index: index + self.sequence_length + 24]

        return values


def load_data(dataset_option: str, save_path: str, sequence_length: int = 72,
              split_sizes: tuple[float, float] = (0.75, 0.25), batch_size: int = 64, shuffle: bool = True,
              drop_last: bool = False, num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    This method returns Dataloaders of the chosen dataset

    Parameters
    ----------
    dataset_option: str
        path for the dataset
    save_path : str
    sequence_length : int
        sequence length for the neural net
    split_sizes : tuple[float, float]
    batch_size: int
    shuffle : bool
    drop_last : bool
    num_workers: int
        num workers for the

    Returns
    -------
    DataLoader
        a Dataloader to iterate through the data
    """

    # create initial datasets
    df_train_val, df_test = read_data(dataset=dataset_option, path=save_path, years_test=2)
    df_test = pd.concat([df_train_val[-sequence_length:], df_test], axis=0)
    means = {}
    stds = {}
    for variable in df_train_val.columns:
        means[variable] = df_train_val[variable].mean()
        stds[variable] = df_train_val[variable].std()
        df_train_val[variable] = (df_train_val[variable] - means[variable]) / stds[variable]

    df_train = df_train_val[:-42 * 7 * 24]
    df_val = df_train_val[-42 * 7 * 24:]
    df_val = pd.concat([df_train[-sequence_length:], df_val], axis=0)
    train_dataset = ElectricDataset(df_train, sequence_length)
    val_dataset = ElectricDataset(df_val, sequence_length, evaluate=True)
    test_dataset = ElectricDataset(df_test, sequence_length, evaluate=True)

    # define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=drop_last,
                                 num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader, means['Price'], stds['Price']


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Parameters
    ----------
    seed : int

    Returns
    -------
    None
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


def benchmark_dnn_without_retrain(datasets: list[str], last_year: bool = False) -> None:
    # define metrics vector
    results = []

    for dataset in datasets:

        df_train, df_test = read_data(f'./data/epftoolbox/{dataset}', dataset=dataset, years_test=2,
                                      begin_test_date=None, end_test_date=None)

        model = DNN(
            experiment_id='1', path_hyperparameter_folder='./epftoolbox/examples/experimental_files', nlayers=2,
            dataset=dataset, years_test=2, shuffle_train=True, data_augmentation=0,
            calibration_window=4)

        forecast_dates = df_test.index[::24]
        forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
        real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
        real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

        # start to calibrate model with train data
        Xtrain, Ytrain, Xval, Yval, Xtest, _, _ = \
            _build_and_split_XYs(dfTrain=df_train, features=model.best_hyperparameters,
                                 shuffle_train=True, dfTest=df_test, date_test=None,
                                 data_augmentation=model.data_augmentation,
                                 n_exogenous_inputs=len(df_train.columns) - 1)

        # Normalizing the input and outputs if needed
        Xtrain, Xval, Xtest, Ytrain, Yval = model._regularize_data(Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain,
                                                                   Yval=Yval)

        model.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval)

        # for loop over the recalibration dates
        for date in forecast_dates:
            # For simulation purposes, we assume that the available data is
            # the data up to current date where the prices of current date are not known
            if last_year:
                data_available = pd.concat([df_train[-365*24:], df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)
            else:
                data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

            # We extract real prices for current date and set them to NaN in the dataframe of available data
            data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

            # Recalibrating the model with the most up-to-date available data and making a prediction
            # for the next day
            Yp = forecast_next_day(model, df=data_available, next_day_date=date)

            # Saving the current prediction
            forecast.loc[date, :] = Yp

        # Computing metrics up-to-current-date
        mae = MAE(real_values.values, forecast.values)
        rmse = RMSE(real_values.values, forecast.values)
        smape = sMAPE(real_values.values, forecast.values)

        # appending results
        results.append([mae, rmse, smape])

        # saving forecast
        if not os.path.isdir(f'./results/forecasts/dnn_ensemble_without_retraining_'
                             f'last_year_{last_year}'):
            os.makedirs(f'./results/forecasts/dnn_ensemble_without_retraining_last_year_{last_year}')
        forecast.to_csv(f'./results/forecasts/dnn_ensemble_without_retraining_last_year_{last_year}/'
                        f'{dataset}.csv')

    # save results
    df = pd.DataFrame(data=np.array(results), index=datasets, columns=['MAE', 'RMSE', 'sMAPE'])
    if not os.path.isdir('./results/benchmarks'):
        os.makedirs('./results/benchmarks')
    df.to_csv(f'./results/benchmarks/dnn_ensemble_without_retraining_last_year_{last_year}.csv')

    return None


def forecast_next_day(model, df, next_day_date):
    """Method that builds an easy-to-use interface for daily recalibration and forecasting of the DNN model
    
    The method receives a pandas dataframe ``df`` and a day ``next_day_date``. Then, it 
    recalibrates the model using data up to the day before ``next_day_date`` and makes a prediction
    for day ``next_day_date``.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of historical data containing prices and N exogenous inputs. The index of the 
        dataframe should be dates with hourly frequency. The columns should have the following 
        names ['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']
    next_day_date : TYPE
        Date of the day-ahead
    
    Returns
    -------
    numpy.array
        An array containing the predictions in the provided date
    
    """

    # We define the new training dataset considering the last calibration_window years of data 
    df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
    df_train = df_train.loc[next_day_date - pd.Timedelta(hours=model.calibration_window * 364 * 24):]

    # We define the test dataset as the next day (they day of interest) plus the last two weeks
    # in order to be able to build the necessary input features.
    df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

    # Generating training, validation, and test input and outpus. For the test dataset,
    # even though the dataframe contains 15 days of data (next day + last 2 weeks),
    # we provide as parameter the date of interest so that Xtest and Ytest only reflect that
    Xtrain, Ytrain, Xval, Yval, Xtest, _, _ = \
        _build_and_split_XYs(dfTrain=df_train, features=model.best_hyperparameters, 
                            shuffle_train=True, dfTest=df_test, date_test=next_day_date,
                            data_augmentation=model.data_augmentation, 
                            n_exogenous_inputs=len(df_train.columns) - 1)

    # Normalizing the input and outputs if needed
    Xtrain, Xval, Xtest, Ytrain, Yval = \
        model._regularize_data(Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain, Yval=Yval)

    # Recalibrating the neural network and extracting the prediction
    Yp = model.predict(Xtest=Xtest)

    return Yp


@torch.no_grad()
def benchmark_naive_daily_model(datasets: list[str]) -> None:
    """
    This function computes a benchmark predicting the next day as the present day values

    Returns
    -------
    None
    """

    # define metrics vector
    results = []

    for dataset in datasets:
        # load test data
        df_train, df_test = read_data(f'{DATASETS_PATH}/{dataset}', dataset=dataset, years_test=2,
                                      begin_test_date=None, end_test_date=None)
        df_test = pd.concat([df_train[-24:], df_test], axis=0)

        # define teste number of days
        test_number_of_days = len(df_test) // 24

        # define forecast object
        real_values = np.zeros((test_number_of_days - 1, 24))
        forecast = np.zeros((test_number_of_days - 1, 24))

        # iter over test days
        for i in range(test_number_of_days - 1):
            real_values[i] = df_test['Price'][24 + 24 * i:48 + 24 * i].to_numpy()
            forecast[i] = df_test['Price'][24 * i:24 * i + 24].to_numpy()

        # computing metrics up-to-current-date
        mae = MAE(real_values, forecast)
        rmse = RMSE(real_values, forecast)
        smape = sMAPE(real_values, forecast)

        # add dataset results
        results.append([mae, rmse, smape])

        # saving forecast
        if not os.path.isdir(f'./results/forecasts/dump_daily_model'):
            os.makedirs(f'./results/forecasts/naive_daily_model')
        df_forecast = pd.DataFrame(index=df_test.index[24::24], columns=['h' + str(k) for k in range(24)],
                                   data=forecast)
        df_forecast.to_csv(f'./results/forecasts/naive_daily_model/{dataset}.csv')

    df = pd.DataFrame(data=np.round_(np.array(results), decimals=3), index=datasets,
                      columns=['MAE', 'RMSE', 'sMAPE'])
    if not os.path.isdir('./results/benchmarks'):
        os.makedirs('./results/benchmarks')
    df.to_csv(f'./results/benchmarks/naive_daily_model.csv')


@torch.no_grad()
def final_results(datasets: list[str]) -> None:
    results = []

    for dataset in datasets:

        # load data
        train_data, val_data, test_data, mean, std = load_data(dataset, f'{DATASETS_PATH}/{dataset}',
                                                               sequence_length=128, batch_size=128)

        directories = os.listdir(f'./best_models/{dataset}')
        model_path = directories[1] if directories[0].split('.')[-1] == 'csv' else directories[0]
        # model_type = directories[0].split('_')[0]
        model = torch.jit.load(f'./best_models/{dataset}/{model_path}')

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

        forecast = pd.read_csv(f'./best_models/{dataset}/forecast.csv')
        forecast = forecast.to_numpy()[:, 1:]

        mae = MAE(real_values, forecast)
        rmse = RMSE(real_values, forecast)
        smape = sMAPE(real_values, forecast)

        forecast_dnn_ensemble = pd.read_csv(f'./results/forecasts/dnn_ensemble_without_retraining_last_'
                                            f'year_False/{dataset}.csv')
        forecast_dnn_ensemble = forecast_dnn_ensemble.to_numpy()[:, 1:]
        dm_test_dnn_ensemble = DM(real_values, forecast_dnn_ensemble, forecast, version='multivariate')
        forecast_dnn_ensemble_last_year = pd.read_csv(f'./results/forecasts/dnn_ensemble_without_'
                                                      f'retraining_last_year_True/{dataset}.csv')
        forecast_dnn_ensemble_last_year = forecast_dnn_ensemble_last_year.to_numpy()[:, 1:]
        dm_test_dnn_ensemble_last_year = DM(real_values, forecast_dnn_ensemble_last_year, forecast,
                                            version='multivariate')
        forecast_naive_model = pd.read_csv(f'./results/forecasts/naive_daily_model/{dataset}.csv')
        forecast_naive_model = forecast_naive_model.to_numpy()[:, 1:]
        dm_test_naive_model = DM(real_values, forecast_naive_model, forecast, version='multivariate')

        results.append([mae, rmse, smape, dm_test_dnn_ensemble, dm_test_dnn_ensemble_last_year, dm_test_naive_model])

    df = pd.DataFrame(data=np.round_(np.array(results), decimals=3), index=datasets,
                      columns=['MAE', 'RMSE', 'sMAPE', 'DM test DNN', 'DM test DNN last year', 'DM test Dum Model'])
    if not os.path.isdir(f'./results'):
        os.makedirs(f'./results')
    df.to_csv(f'./results/results.csv')


def draw_forecasts(datasets: list[str], save_path: str, number_per_day: int = 2) -> None:
    """
    This function creates visualizations of the forecasts of the models
    Parameters
    ----------
    datasets: list[str]
        list of datasets
    number_per_day : int
        number of visualizations per day of the week

    Returns
    -------
    None
    """

    for dataset in datasets:
        # define vector for number of visualizations
        days_of_week = [0, 0, 0, 0, 0, 0, 0]

        # load model
        files = os.listdir(f'./best_models/epftoolbox/{dataset}')
        model = torch.jit.load(f'./best_models/epftoolbox/{dataset}/{files[0]}')

        # load variables
        model_class = files[0].split('_')[0]
        sequence_length = int(files[0].split('sl_')[-1].split('_')[0])

        # load data
        train_data, val_data, test_data, mean, std = load_data(dataset, f'./data/epftoolbox/{dataset}',
                                                               sequence_length=sequence_length, batch_size=128)
        df_train, df_test = read_data(dataset=dataset, path=f'{DATASETS_PATH}/{dataset}', years_test=2)

        i = 0
        for inputs_test in test_data:
            # prepare data
            inputs_test = inputs_test.float()
            values = inputs_test[:, :-24, 0].unsqueeze(2)
            features = inputs_test[:, 24:, 1:3]
            hours = inputs_test[:, 24:, 3].long()
            days = inputs_test[:, 24:, 4].long()
            months = inputs_test[:, 24:, 5].long() - 1
            targets = inputs_test[:, -24:, 0]

            # update train with new values
            df_train = pd.concat([df_train[-365 * 24:], df_test[:24 * i]], axis=0)

            # compute new mean and std
            means = {}
            stds = {}
            for variable in df_train.columns:
                means[variable] = df_train[variable].mean()
                stds[variable] = df_train[variable].std()

            values = (values - means['Price']) / stds['Price']
            features[:, :, 0] = (features[:, :, 0] - means['Exogenous 1']) / stds['Exogenous 1']
            features[:, :, 1] = (features[:, :, 1] - means['Exogenous 2']) / stds['Exogenous 2']

            # compute output and add forecast and real values
            if model_class == 'base' or model_class == 'base_daily':
                outputs = model(values, features)[:, -24:]
            else:
                outputs = model(values, features, hours, days, months)

            values = values * stds['Price'] + means['Price']
            past_values = values[0, :, 0].detach().cpu().numpy()
            real_values = targets[0].detach().cpu().numpy()
            real_values = np.concatenate((past_values, real_values))
            forecast = outputs[0].detach().cpu().numpy() * stds['Price'] + means['Price']
            # forecast = np.concatenate((past_values, forecast))

            if days_of_week[days[0, -1].item()] < number_per_day:
                plt.figure()
                plt.plot(np.arange(len(real_values)), real_values)
                plt.plot(np.arange(len(past_values), len(real_values)), forecast)
                plt.xlabel('time [hour]')
                plt.ylabel('Price [EUR/MWh]')
                plt.legend(['Real values', 'Forecast'])

                # save and close figure
                if not os.path.isdir(f'{save_path}/{dataset}/{days[0, -1].item()}'):
                    os.makedirs(f'{save_path}/{dataset}/{days[0, -1].item()}')
                plt.savefig(f'{save_path}/{dataset}/{days[0, -1].item()}/{days_of_week[days[0, -1].item()]}.pdf')
                plt.close()

                days_of_week[days[0, -1].item()] += 1

            i += 1

    return None


if __name__ == '__main__':
    main()
