# Solar Power Forcasting

## Environment Installation

- Ubuntu 18.04

```
conda env create -f environment.yml -n <env_name>
```

## Run Code Step by Step

1. Produce Training Data, Apply CWT and PCA.

```
python process_data.py
```

2. Train Model to Predict Power

- TPA-LSTM

```
python train_tpalstm.py
```

- TCN

```
python train_tcn.py
```

- XGBoost

```
python train_xgboost.py
``` 
- NBeats

```
python train_nbeats.py
```

## Do

1. 更改timesteps長度

2. 使用殘差資料訓練，預測發電變化量