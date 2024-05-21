# %%
import os
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , StandardScaler , LabelEncoder , OneHotEncoder 
from pytorch_forecasting import TimeSeriesDataSet , GroupNormalizer , Baseline, DeepAR
from pytorch_forecasting.metrics import QuantileLoss , MAE,SMAPE, MultiHorizonMetric,DistributionLoss , NormalDistributionLoss, MultivariateNormalDistributionLoss
from pytorch_forecasting.data import NaNLabelEncoder 
from pytorch_forecasting.data.encoders import TorchNormalizer , EncoderNormalizer
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# %%
# Check if CUDA is available
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(torch.cuda.current_device()))


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
    max_encoder_length=max_encoder_length,
    min_prediction_length=14,
    max_prediction_length=max_prediction_length,
    static_categoricals=['LCLid','EnergyClass', 'Acorn_grouped'],
    time_varying_known_categoricals=["holiday"],
    time_varying_known_reals=['temperature_avg',"day_of_week", "month_of_year"],
    time_varying_unknown_reals=['energy_sum'],
    categorical_encoders={"LCLid": NaNLabelEncoder().fit(df.LCLid)},
    target_normalizer=EncoderNormalizer(transformation="log1p"), 
    
)


validation = TimeSeriesDataSet.from_dataset(training, df[lambda x: x.days_from_start <= validation_cutoff], predict=True, stop_randomization=True)

test = TimeSeriesDataSet.from_dataset(
    training, 
    df,
    predict=True, 
    stop_randomization=True
)




# %%
# create dataloaders for  our model
batch_size = 128
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=5, batch_sampler="synchronized")
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=5, batch_sampler="synchronized")
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=5, batch_sampler="synchronized")

# %%

torch.set_float32_matmul_precision('high') 


# %%
#Baseline model
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)]).to("cuda")
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()


# %%
torch.manual_seed(64)
early_stop_callback = EarlyStopping(monitor="val_loss", patience = 5, min_delta=1e-4, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  
checkpoint = ModelCheckpoint( save_top_k=2, monitor="val_loss")
logger = TensorBoardLogger("lightning_logs_DeepAR", log_graph= True) 

trainer = pl.Trainer(
    max_epochs=10,
    accelerator='gpu', 
    # devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    enable_progress_bar=False,
    # accumulate_grad_batches=2,  # Or higher, depending on your needs
    profiler="simple",
    # precision="16-mixed",  # Enable mixed precision training
    callbacks=[lr_logger, early_stop_callback,checkpoint],
    logger=logger)

net = DeepAR.from_dataset(
    training,
    learning_rate=0.003,
    log_interval=10,
    log_val_interval=1,
    hidden_size=16,
    rnn_layers=3,
    optimizer='Adam',  # Specifying the optimizer type 
    # optimizer_params={'weight_decay': 1e-3},
    dropout = 0.2,
    loss=NormalDistributionLoss(),
    
)



# %%
trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader)

# %%
val_results = trainer.validate(model = net, dataloaders = val_dataloader , ckpt_path = 'best')

# Test using the best model checkpoint
results = trainer.test(model=net, dataloaders=test_dataloader, ckpt_path="best")



