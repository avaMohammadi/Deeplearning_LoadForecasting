import optuna
from pytorch_lightning import Callback
import os
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , StandardScaler , LabelEncoder , OneHotEncoder 
from pytorch_forecasting import TimeSeriesDataSet , GroupNormalizer , Baseline , RecurrentNetwork
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor , ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.models.nn.rnn import LSTM , GRU


# %%
data = pd.read_csv('filtered_data.csv')


# %%
data['Acorn_grouped'].unique()


# %%
data[['LCLid','ToU','holiday']] = data[['LCLid','ToU','holiday']].astype('object')




# %%
df = data[['LCLid', 'energy_sum','temperature_avg','EnergyClass', 'Acorn_grouped','days_from_start','day_of_week', 'month_of_year','holiday']] 


# %%

max_prediction_length = 14
max_encoder_length = 12*14

training_cutoff = df["days_from_start"].max() - (max_prediction_length * 2)  
validation_cutoff = df["days_from_start"].max() - max_prediction_length 

training = TimeSeriesDataSet(
    df[lambda x: x.days_from_start <= training_cutoff],
    time_idx="days_from_start",
    target="energy_sum",
    group_ids=["LCLid"],
    min_encoder_length=max_encoder_length // 2, 
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["LCLid",'EnergyClass', 'Acorn_grouped'],
    time_varying_known_categoricals=["holiday"],
    time_varying_known_reals=['temperature_avg',"day_of_week", "month_of_year"],
    time_varying_unknown_reals=['energy_sum'],
    target_normalizer=GroupNormalizer(
        groups=["LCLid"], transformation="softplus"
    ),  # we normalize by group
    add_relative_time_idx=True,
    #add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df[lambda x: x.days_from_start <= validation_cutoff], predict=True, stop_randomization=True)


test = TimeSeriesDataSet.from_dataset(
    training, 
    df,
    predict=True, 
    stop_randomization=True
)


torch.set_float32_matmul_precision('high')

# %%
# create dataloaders for  our model
batch_size = 256
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=5, batch_sampler="synchronized")
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=5, batch_sampler="synchronized")
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=5, batch_sampler="synchronized")



# Specify the SQLite URL for persistent storage, or use an in-memory storage for temporary studies.
storage_url = "sqlite:///lstm_hyperparameter_tuning.db"
study = optuna.create_study(study_name='LSTM_hyperparameter_optimization', direction='minimize', storage=storage_url, load_if_exists=True)



def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    rnn_layers = trial.suggest_int("rnn_layers", 1, 3)
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Setup model
    lstm_model = RecurrentNetwork.from_dataset(
        training,
        cell_type="LSTM",
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        learning_rate=learning_rate,
        # log_interval=10,
        loss=MAE(),
        optimizer="Adam",
        reduce_on_plateau_patience=4,
        dropout=dropout
    )

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, verbose=False, mode="min")],
        logger=False,  # Disable logging for optimization
       # checkpoint_callback=False  # Disable model checkpointing
    )

    # Train the model
    trainer.fit(lstm_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Capture the best validation loss
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss



study.optimize(objective, n_trials=30)  # Adjust n_trials and timeout as needed

# Display the best trial info
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


