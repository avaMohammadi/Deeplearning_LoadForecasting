# %%
import os
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , StandardScaler , LabelEncoder , OneHotEncoder 
from pytorch_forecasting import TimeSeriesDataSet , GroupNormalizer , Baseline , RecurrentNetwork
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.models.nn.rnn import LSTM , GRU



# %%
data = pd.read_csv('filtered_data.csv')


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
    add_target_scales=True,
    add_encoder_length=True,
)



validation = TimeSeriesDataSet.from_dataset(training, df[lambda x: x.days_from_start <= validation_cutoff], predict=True, stop_randomization=True)


test = TimeSeriesDataSet.from_dataset(
    training, 
    df,
    predict=True, 
    stop_randomization=True
)

# create dataloaders for  our model
batch_size = 64
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=5)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=5)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=5)


# %%

torch.set_float32_matmul_precision('high') 


# %%
#Baseline model
actuals = torch.cat([y for x, (y, weight) in iter(test_dataloader)]).to("cuda")
baseline_predictions = Baseline().predict(test_dataloader)
result = (actuals - baseline_predictions).abs().mean().item()

print(f"The baseline result is: {result}")


# %%
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=2, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  
logger = TensorBoardLogger("lightning_logs_lstm")  

trainer = pl.Trainer(
    max_epochs=10,
    accelerator='gpu', 
    # devices=1,
    enable_model_summary=True,
    enable_progress_bar=False,
    gradient_clip_val=0.1,
    # accumulate_grad_batches=2,  
    profiler="simple",
    # precision="16-mixed",  # Enable mixed precision training
    callbacks=[lr_logger, early_stop_callback],
    logger=logger)


lstm_model = RecurrentNetwork.from_dataset(
    training,
    cell_type="LSTM",  
    hidden_size=16,  
    rnn_layers=2, 
    learning_rate=0.0006,
    log_interval=10,
    loss=MAE(),
    reduce_on_plateau_patience=4,
    dropout=0.3, 
)




# %%
trainer.fit(
    lstm_model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader)

# %%
val_results = trainer.validate(model = lstm_model, dataloaders = val_dataloader , ckpt_path = 'best')

# Test using the best model checkpoint
results = trainer.test(model=lstm_model, dataloaders=test_dataloader, ckpt_path="best")


