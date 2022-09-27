# deep learning libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from epftoolbox.evaluation import MAE, RMSE, MAPE, sMAPE, DM

# own modules
from src.utils import ElectricDataset


@torch.enable_grad()
def train(train_data: DataLoader, val_data: DataLoader, model: torch.nn.Module, model_class: str,
          mean: float, std: float, loss: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: object,
          clip_gradients: float, epochs: int, writer: SummaryWriter, save_path: str) -> None:

    for epoch in range(epochs):

        # print epoch
        print(epoch)

        # train
        model.train()

        # initialize loss vectors
        mae_vector = []
        rmse_vector = []
        mape_vector = []
        smape_vector = []
        losses = []

        # train loop
        for inputs in train_data:
            # prepare data
            inputs = inputs.float()
            values = inputs[:, :-24, 0].unsqueeze(2)
            features = inputs[:, 24:, 1:3]
            hours = inputs[:, 24:, 3].long()
            days = inputs[:, 24:, 4].long()
            months = inputs[:, 24:, 5].long() - 1
            targets = inputs[:, 24:, 0]

            # compute output and loss
            if model_class == 'base' or model_class == 'base_daily':
                outputs = model(values, features)
            else:
                outputs = model(values, features, hours, days, months)
            loss_value = loss(outputs, targets)

            # backward and optimize
            optimizer.zero_grad()
            loss_value.backward()
            if clip_gradients is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
            optimizer.step()

            # add loss and mae for the step
            losses.append(loss_value.item())
            mae_vector.append(MAE(targets.detach().cpu().numpy() * std + mean,
                                  outputs.detach().cpu().numpy() * std + mean))
            rmse_vector.append(RMSE(targets.detach().cpu().numpy() * std + mean,
                                    outputs.detach().cpu().numpy() * std + mean))
            mape_vector.append(MAPE(targets.detach().cpu().numpy() * std + mean,
                                    outputs.detach().cpu().numpy() * std + mean))
            smape_vector.append(sMAPE(targets.detach().cpu().numpy() * std + mean,
                                      outputs.detach().cpu().numpy() * std + mean))

        # add measures to tensorboard
        writer.add_scalar('loss', np.mean(losses), epoch)
        writer.add_scalar('MAE/train', np.mean(mae_vector), epoch)
        writer.add_scalar('RMSE/train', np.mean(rmse_vector), epoch)
        writer.add_scalar('MAPE/train', np.mean(mape_vector), epoch)
        writer.add_scalar('sMAPE/train', np.mean(smape_vector), epoch)
        if optimizer.param_groups[0]['lr'] is not None:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # update scheduler
        if scheduler is not None:
            scheduler.step()

        # evaluate
        model.eval()
        with torch.no_grad():

            # initialize losses vectors
            mae_vector = []
            rmse_vector = []
            mape_vector = []
            smape_vector = []

            # val loop
            for inputs in val_data:
                # prepare data
                inputs = inputs.float()
                values = inputs[:, :-24, 0].unsqueeze(2)
                features = inputs[:, 24:, 1:3]
                hours = inputs[:, 24:, 3].long()
                days = inputs[:, 24:, 4].long()
                months = inputs[:, 24:, 5].long() - 1
                targets = inputs[:, -24:, 0]

                # compute output and loss
                if model_class == 'base' or model_class == 'base_daily':
                    outputs = model(values, features)[:, -24:]
                else:
                    outputs = model(values, features, hours, days, months)

                # add loss and mae for the step
                mae_vector.append(MAE(targets.detach().cpu().numpy() * std + mean,
                                      outputs.detach().cpu().numpy() * std + mean))
                rmse_vector.append(RMSE(targets.detach().cpu().numpy() * std + mean,
                                        outputs.detach().cpu().numpy() * std + mean))
                mape_vector.append(MAPE(targets.detach().cpu().numpy() * std + mean,
                                        outputs.detach().cpu().numpy() * std + mean))
                smape_vector.append(sMAPE(targets.detach().cpu().numpy() * std + mean,
                                          outputs.detach().cpu().numpy() * std + mean))

            # add measures to tensorboard
            writer.add_scalar('MAE/val', np.mean(mae_vector), epoch)
            writer.add_scalar('RMSE/val', np.mean(rmse_vector), epoch)
            writer.add_scalar('MAPE/val', np.mean(mape_vector), epoch)
            writer.add_scalar('sMAPE/val', np.mean(smape_vector), epoch)

    # save model
    torch.jit.script(model).save(f'{save_path}')


@torch.no_grad()
def test(df_train: pd.DataFrame, df_test: pd.DataFrame, test_data: DataLoader, model_class: str, load_path: str,
         compare_paths: list[str], save: bool, save_path: str, name: str) -> None:

    # load model
    model = torch.jit.load(f'{load_path}')

    # evaluate
    model.eval()

    # initialize losses vectors
    real_values = torch.zeros((test_data.__len__(), 24))
    forecast = torch.zeros((test_data.__len__(), 24))

    # test loop
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

        real_values[i] = targets.detach().cpu()
        forecast[i] = outputs.detach().cpu() * stds['Price'] + means['Price']

        i += 1

    # compute measures
    real_values = real_values.numpy()
    forecast = forecast.numpy()
    mae = MAE(real_values, forecast)
    rmse = RMSE(real_values, forecast)
    mape = MAPE(real_values, forecast)
    smape = sMAPE(real_values, forecast)

    # save forecast
    df_forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)],
                               data=forecast)
    if save:
        df_forecast.to_csv(f'{save_path}/forecast.csv')

    print(f'MAE: {mae:.3f}  |  RMSE: {rmse:.3f} | MAPE: {mape:.3f} | sMAPE: {smape:.3f}')
    i = 1
    for compare_path in compare_paths:
        forecast_dnn_ensemble = pd.read_csv(f'{compare_path}')
        forecast_dnn_ensemble = forecast_dnn_ensemble.to_numpy()[:, 1:]
        dm_test = DM(real_values, forecast_dnn_ensemble, forecast, version='multivariate')
        print(f'DM test {i}: {dm_test}')
        i += 1

    # save model
    if save:
        torch.jit.script(model).save(f'{save_path}/{name}.pt')

    return None
