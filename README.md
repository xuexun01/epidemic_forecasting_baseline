## Baselines for Epidemic Forecasting

This project includes some basic models for reproducing infectious disease prediction, so that subsequent methods can use these models to verify the effects when doing performance comparison experiments.
Baselines includes the following methods:

* ARIMA
* LSTM 
* CNNRNN_Res 
* STGCN [(https://github.com/VeritasYin/STGCN_IJCAI-18)](https://github.com/VeritasYin/STGCN_IJCAI-18)
* ColaGNN [(https://github.com/amy-deng/colagnn)](https://github.com/amy-deng/colagnn)
* EpiGNN [(https://github.com/Xiefeng69/EpiGNN)](https://github.com/Xiefeng69/EpiGNN)
* STAN  (STAN: Spatio-Temporal Attention Network for Pandemic Prediction Using Real World Evidence)
* MepoGNN [(https://github.com/deepkashiwa20/MepoGNN)](https://github.com/deepkashiwa20/MepoGNN)

>Note: The STAN work has some errors in the data processing part of its own dataset. Specifically, when dealing with missing data, they use the data of another column to cover the data of the missing column, which introduces errors. Therefore, it is not recommended to reproduce this work on the paper dataset.