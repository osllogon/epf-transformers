# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from epftoolbox.data import read_data
from transformers.optimization import Adafactor, AdafactorSchedule

# other libraries
import os
from typing import Optional

# own modules
from src.utils import set_seed, load_data
from src.models import BaseDailyElectricTransformer, sMAPELoss
from src.train_functions import train, test

# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set all seeds
set_seed(42)

# static variables
DATASETS_PATH = './data/epftoolbox'
DATASET = 'FR'

def main() -> None:
    # check device
    print(device)

    # execution mode
    exec_mode = 'test'

    # hyperparameters
    epochs = 100
    model_class = 'base_daily'
    sequence_length = 336
    lr = 1e-5
    num_layers = 6
    num_heads = 8
    embedding_dim = 512
    dim_feedforward = 1024
    normalize_first = False
    dropout = 0.2
    activation = 'relu'
    loss_name = 'mae'
    optimizer_name = 'adamw'
    weight_decay = 1e-2
    clip_gradients = 0.25
    scheduler_name = 'steplr_50_0.2'
    save_model = True

    # load data
    train_dataloader, val_dataloader, test_dataloader, mean, std = load_data(DATASET, f'{DATASETS_PATH}/{DATASET}',
                                                                             sequence_length=sequence_length,
                                                                             batch_size=128)
    df_train, df_test = read_data(dataset=DATASET, path=f'{DATASETS_PATH}/{DATASET}', years_test=2)
    means = {}
    stds = {}
    for variable in df_train.columns:
        means[variable] = df_train[variable].mean()
        stds[variable] = df_train[variable].std()

    # define model name and create tensorboard writer
    name = f'{model_class}_ed_{embedding_dim}_nh_{num_heads}_df_{dim_feedforward}_nl_{num_layers}_sl_' \
           f'{sequence_length}_{normalize_first}_d_{dropout}_a_{activation}_l_{loss_name}_o_{optimizer_name}_' \
           f'lr_{lr}_wd_{weight_decay}_cg_{clip_gradients}_s_{scheduler_name}_e_{epochs}'
    writer = SummaryWriter(f'./runs/epftoolbox/{DATASET}/{name}')

    # define model
    model: torch.nn.Module
    if model_class == 'base_daily':
        model = BaseDailyElectricTransformer(embedding_dim=embedding_dim, num_heads=num_heads,
                                             dim_feedforward=dim_feedforward, num_layers=num_layers,
                                             normalize_first=normalize_first, dropout=dropout,
                                             activation=activation).to(device)
    else:
        raise ValueError('Invalid model class name')

    # define loss
    loss: torch.nn.Module
    if loss_name == 'mse':
        loss = torch.nn.MSELoss()
    elif loss_name == 'mae':
        loss = torch.nn.L1Loss()
    elif loss_name == 'smooth_mae':
        loss = torch.nn.SmoothL1Loss()
    elif loss_name == 'sMAPE':
        loss = sMAPELoss()
    else:
        raise ValueError('Invalid loss name')

    # define optimizer
    optimizer: torch.optim.Optimizer
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adafactor_no_warmup':
        Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=lr,
                  weight_decay=weight_decay, clip_threshold=clip_gradients)
    elif optimizer_name == 'Adafactor':
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=lr,
                              weight_decay=weight_decay, clip_threshold=clip_gradients)
    else:
        raise ValueError('Invalid optimizer name')

    # define scheduler
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    if scheduler_name is not None:
        scheduler_name_pieces = scheduler_name.split('_')
        if scheduler_name_pieces[0] == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(scheduler_name_pieces[1]),
                                                        gamma=float(scheduler_name_pieces[2]))
        elif scheduler_name_pieces[0] == 'multisteplr':
            milestones = [int(piece) for piece in scheduler_name_pieces[1:-1]]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                                             gamma=float(scheduler_name_pieces[-1]))
        elif scheduler_name_pieces[0] == 'adafactorschedule':
            scheduler = AdafactorSchedule(optimizer)
        else:
            raise ValueError('Invalid scheduler name')
    else:
        scheduler = None

    # create dirs is they do not exist
    if not os.path.isdir(f'./models/epftoolbox/{DATASET}'):
        os.makedirs(f'./models/epftoolbox/{DATASET}')
    if not os.path.isdir(f'./best_models/epftoolbox/{DATASET}'):
        os.makedirs(f'./best_models/epftoolbox/{DATASET}')

    # define path for saving models and loading models
    model_path = f'./models/epftoolbox/{DATASET}/{name}.pt'

    # define path for the other model for comparing
    compare_paths = [
        f'./results/epftoolbox/forecasts/dnn_ensemble_without_retraining_last_year_False/{DATASET}.csv',
        f'./results/epftoolbox/forecasts/naive_daily_model/{DATASET}.csv'
    ]

    if exec_mode == 'train':
        # run train function
        train(train_dataloader, val_dataloader, model, model_class, mean, std, loss, optimizer, scheduler,
              clip_gradients, epochs, writer, model_path)

    elif exec_mode == 'test':
        # create dir if it does not exist
        if not os.path.isdir(f'./best_models/epftoolbox/{DATASET}'):
            os.makedirs(f'./best_models/epftoolbox/{DATASET}')

        # delete past best model
        if save_model:
            files = os.listdir(f'./best_models/epftoolbox/{DATASET}')
            if len(files) != 0:
                for file in files:
                    os.remove(f'./best_models/epftoolbox/{DATASET}/{file}')

        # define new path for best model
        best_model_path = f'./best_models/epftoolbox/{DATASET}'

        # run test function
        test(df_train, df_test, test_dataloader, model_class, model_path, compare_paths, save_model, best_model_path, 
             name)

    else:
        raise ValueError('Invalid execution mode')
    
    
if __name__ == '__main__':
    main()
