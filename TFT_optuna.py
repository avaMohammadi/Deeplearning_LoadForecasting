# %%
import os
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , StandardScaler , LabelEncoder , OneHotEncoder 
from pytorch_forecasting import TimeSeriesDataSet , GroupNormalizer , Baseline , QuantileLoss , MAE, MultiHorizonMetric
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import optuna.visualization.matplotlib as ov_mpl
from optuna.pruners import MedianPruner




# %%
data = pd.read_csv('filtered_data.csv')


# %%
data[['LCLid','ToU','holiday']] = data[['LCLid','ToU','holiday']].astype('object')


# %%
df = data[['LCLid', 'energy_sum','temperature_avg','EnergyClass','Acorn_grouped','days_from_start','day_of_week', 'month_of_year','holiday']]



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
    # add_target_scales=True,
    add_encoder_length=True,
)


# validation = TimeSeriesDataSet.from_dataset(training, subset_df, predict=True, stop_randomization=True)
validation = TimeSeriesDataSet.from_dataset(training, df[lambda x: x.days_from_start <= validation_cutoff], predict=True, stop_randomization=True)


test = TimeSeriesDataSet.from_dataset(
    training, 
    df,
    predict=True, 
    stop_randomization=True
)


# %%
# create dataloaders for  our model
batch_size = 256
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=5)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=5)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=5)

# %%
torch.set_float32_matmul_precision('high')

# %%
# Example using SQLite file-based storage
storage_url = "sqlite:///optuna_study_01.db"
study = optuna.create_study(study_name="optuna_study_01", storage=storage_url, load_if_exists=True)



# Define early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss', 
    patience=3,  
    verbose=True,
    min_delta=0.001,
    mode='min',  
)

# Pass the callback to trainer_kwargs
trainer_kwargs = {
    # "limit_train_batches": 0.5,
    "callbacks": [early_stopping_callback],
}

study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test_01",
    n_trials=30,
    max_epochs=10,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(16, 128),
    hidden_continuous_size_range=(8, 64),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=trainer_kwargs,  
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,
    study=study,
    pruner = MedianPruner()
)

print("Best trial:", study.best_trial.params)


