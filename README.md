# Deep Learning for Time-Series Forecasting with Exogenous Variables in Energy Consumption: A Performance and Interpretability Analysis

## Project Overview
This repository contains the code and data for the paper titled "Deep Learning for Time-Series Forecasting with Exogenous Variables in Energy Consumption: A Performance and Interpretability Analysis." The study evaluates different deep learning architectures for short-term load forecasting (STLF) in smart grids, focusing on their ability to integrate exogenous variables for improved predictive accuracy and interpretability.

## Language
- Python

## Algorithms and Models
- Long Short-Term Memory (LSTM)
- [Deep Autoregressive (DeepAR)](https://arxiv.org/abs/1704.04110).
- [Temporal Fusion Transformer (TFT)](https://arxiv.org/abs/1912.09363).
- [TimeLLm](https://arxiv.org/abs/2310.01728).

## Libraries and Tools
- Data Analysis: pandas, numpy, statsmodels
- Machine Learning: keras, TensorFlow, PyTorch, PyTorch Lightning, PyTorch Forecasting, gluonts
- Model Interpretability: SHAP, Feature Permutation, Attention-based Analysis
- Hyperparameter Optimization: Optuna
- Data Visualization: matplotlib, seaborn, TensorBoard
- Other Tools: scikit-learn
- Data Processing & Analysis: pandas, numpy, statsmodels
  
## Dataset
SmartMeter Energy Consumption Data in London Households
-Original Dataset: https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households
-Refactorized Dataset : https://www.kaggle.com/jeanmidev/smart-meters-in-london


