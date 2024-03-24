# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from epftoolbox.data import read_data
from transformers.optimization import Adafactor, AdafactorSchedule

# other libraries
import os
from typing import Optional, Literal, List

# own modules
from src.utils import set_seed, load_data
from src.models import BaseDailyElectricTransformer, sMAPELoss
from src.train_functions import train, test

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds
set_seed(42)

# static variables
DATASETS_PATH: str = "data"


def main() -> None:
    """
    This function is the main function of the train module

    Raises:
        ValueError: Invalid loss name
        ValueError: Invalid optimizer name
        ValueError: Invalid scheduler name
        ValueError: Invalid execution mode
    """

    # variables
    dataset: Literal["NP", "PJM", "FR", "BE", "DE"] = "NP"
    exec_mode: Literal["train", "test"] = "train"
    save_model = True

    # hyperparameters
    epochs = 100
    sequence_length = 336
    lr = 1e-4
    num_layers = 4
    num_heads = 4
    embedding_dim = 128
    dim_feedforward = 1048
    normalize_first = False
    dropout = 0.2
    activation = "relu"
    loss_name = "mae"
    optimizer_name = "adamw"
    weight_decay = 1e-2
    clip_gradients = 0.1
    scheduler_name = "steplr_70_0.1"
    
    # check device
    print(device)

    # load data
    train_dataloader, val_dataloader, test_dataloader, mean, std = load_data(
        dataset,
        f"{DATASETS_PATH}/{dataset}",
        sequence_length=sequence_length,
        batch_size=128,
    )
    df_train, df_test = read_data(
        dataset=dataset, path=f"{DATASETS_PATH}/{dataset}", years_test=2
    )
    means = {}
    stds = {}
    for variable in df_train.columns:
        means[variable] = df_train[variable].mean()
        stds[variable] = df_train[variable].std()

    # define model name and create tensorboard writer
    name: str = (
        f"m_ed_{embedding_dim}_nh_{num_heads}_df_{dim_feedforward}_nl_{num_layers}_sl_"
        f"{sequence_length}_{normalize_first}_d_{dropout}_a_{activation}_l_{loss_name}_o_{optimizer_name}_"
        f"lr_{lr}_wd_{weight_decay}_cg_{clip_gradients}_s_{scheduler_name}_e_{epochs}"
    )
    writer: SummaryWriter = SummaryWriter(f"runs/{dataset}/{name}")

    # define model
    model: torch.nn.Module = BaseDailyElectricTransformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        normalize_first=normalize_first,
        dropout=dropout,
        activation=activation,
    ).to(device)

    # define loss
    loss: torch.nn.Module
    if loss_name == "mse":
        loss = torch.nn.MSELoss()
    elif loss_name == "mae":
        loss = torch.nn.L1Loss()
    elif loss_name == "smooth_mae":
        loss = torch.nn.SmoothL1Loss()
    elif loss_name == "sMAPE":
        loss = sMAPELoss()
    else:
        raise ValueError("Invalid loss name")

    # define optimizer
    optimizer: torch.optim.Optimizer
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "Adafactor_no_warmup":
        Adafactor(
            model.parameters(),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=lr,
            weight_decay=weight_decay,
            clip_threshold=clip_gradients,
        )
    elif optimizer_name == "Adafactor":
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=lr,
            weight_decay=weight_decay,
            clip_threshold=clip_gradients,
        )
    else:
        raise ValueError("Invalid optimizer name")

    # define scheduler
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    if scheduler_name is not None:
        scheduler_name_pieces = scheduler_name.split("_")
        if scheduler_name_pieces[0] == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_name_pieces[1]),
                gamma=float(scheduler_name_pieces[2]),
            )
        elif scheduler_name_pieces[0] == "multisteplr":
            milestones = [int(piece) for piece in scheduler_name_pieces[1:-1]]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=float(scheduler_name_pieces[-1])
            )
        elif scheduler_name_pieces[0] == "adafactorschedule":
            scheduler = AdafactorSchedule(optimizer)
        else:
            raise ValueError("Invalid scheduler name")
    else:
        scheduler = None

    # create dirs is they do not exist
    if not os.path.isdir(f"models/{dataset}"):
        os.makedirs(f"models/{dataset}")
    if not os.path.isdir(f"best_models/{dataset}"):
        os.makedirs(f"best_models/{dataset}")

    # define path for saving models and loading models
    model_path: str = f"models/{dataset}/{name}.pt"

    # define path for the other model for comparing
    compare_paths: List[str] = [
        f"results/forecasts/dnn_ensemble_without_retraining_last_year_False/{dataset}.csv",
        f"results/forecasts/naive_daily_model/{dataset}.csv",
    ]

    if exec_mode == "train":
        # run train function
        train(
            train_dataloader,
            val_dataloader,
            model,
            mean,
            std,
            loss,
            optimizer,
            scheduler,
            clip_gradients,
            epochs,
            writer,
            model_path,
        )

    elif exec_mode == "test":
        # create dir if it does not exist
        if not os.path.isdir(f"best_models/{dataset}"):
            os.makedirs(f"best_models/{dataset}")

        # delete past best model
        if save_model:
            files = os.listdir(f"best_models/{dataset}")
            if len(files) != 0:
                for file in files:
                    os.remove(f"best_models/{dataset}/{file}")

        # define new path for best model
        best_model_path: str = f"best_models/{dataset}"

        # run test function
        test(
            df_train,
            df_test,
            test_dataloader,
            model_path,
            compare_paths,
            save_model,
            best_model_path,
            name,
        )

    else:
        raise ValueError("Invalid execution mode")


if __name__ == "__main__":
    main()
