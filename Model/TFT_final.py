# %%
import os
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , StandardScaler , LabelEncoder , OneHotEncoder 
from pytorch_forecasting import TimeSeriesDataSet , GroupNormalizer , Baseline , QuantileLoss , MAE, MultiHorizonMetric
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger



# %%
data = pd.read_csv('filtered_data.csv')


# %%
data[['LCLid','ToU','holiday']] = data[['LCLid','ToU','holiday']].astype('object')


# %%
df = data[['LCLid', 'energy_sum','temperature_avg','EnergyClass', 'Acorn_grouped','days_from_start','day_of_week', 'month_of_year','holiday']]



# %%

max_prediction_length = 14
max_encoder_length = 12*14

training_cutoff = df["days_from_start"].max() - (max_prediction_length * 2)  # Leaving space for validation and test
validation_cutoff = df["days_from_start"].max() - max_prediction_length  # Last part for testing

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
batch_size = 128
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
logger = TensorBoardLogger("lightning_logs")  

trainer = pl.Trainer(
    max_epochs=20,
    accelerator='gpu', 
    # devices=1,
    enable_model_summary=True,
    enable_progress_bar=False,
    gradient_clip_val=0.2,
    # accumulate_grad_batches=2,  
    profiler="simple",
    # precision="16-mixed",
    callbacks=[lr_logger, early_stop_callback],
    logger=logger)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.2,
    hidden_continuous_size=16,
    # output_size=7,  
    loss=QuantileLoss(),
    optimizer="Ranger",
    log_interval=10, 
    reduce_on_plateau_patience=4)


# %%
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader)

# %%
trainer.validate(model = tft , ckpt_path= 'best', dataloaders= val_dataloader)
trainer.test(model = tft , ckpt_path= 'best', dataloaders= test_dataloader)

