import pandas as pd
import numpy as np
import reservoirpy as rpy

from foreutils.denoising import (
    denoise_time_series,
    downsample_time_series,
    moving_std_filter,
    holt_winters_filter,
)
from foreutils.utils import create_training_data
from foreutils.forecasting import Forecaster


def fault_detection(
    signal: pd.Series,
    ESN_signal: rpy.model.Model,
    ESN_residuals: rpy.model.Model,
    forecasted_steps: int = 10,
    residuals_training_steps: int = 208,
    k: int = 1,
    h: int = 1,
    N: int = 4,
) -> bool:

    """
    Detect if a forecasted NDVI signal diverges from a real signal measure.

    Parameters:
    - signal (pd.Series): The input NDVI signal.
    - ESN_signal (rpy.model.Model): Echo State Network model for signal prediction.
    - ESN_residuals (rpy.model.Model): Echo State Network model for residuals prediction.
    - forecasted_steps (int, optional): Number of future steps to forecast. Default is 10.
    - residuals_training_steps (int, optional): Number of steps to train the residuals. Default is 208.
    - k (int, optional): Scaling factor for residuals forecast. Default is 1.
    - h (int, optional): Iteration step size. Default is 1.
    - N (int, optional): Length of forecast segment to check against lower bound. Default is 4.

    Returns:
    - bool: True if forecasted signal diverges from the actual signal, otherwise False.

    Procedure:
    1. Denoise the input signal.
    2. Partition the denoised signal into training data.
    3. Forecast the denoised signal using the ESN_signal model.
    4. Compute the residuals between actual and forecasted signals.
    5. Forecast the residuals using the ESN_residuals model.
    6. Compute the lower bound for fault detection.
    7. Compare forecasted signal segments against the lower bound to detect faults.

    """

    # Denoise signal
    denoised_signal_series = denoise_time_series(
        signal, [downsample_time_series, moving_std_filter, holt_winters_filter]
    )

    # Parameters setup
    denoised_signal = denoised_signal_series.to_numpy()
    X, y = create_training_data(denoised_signal, num_features=52)

    T = len(X)
    s = T - forecasted_steps
    s_tilde = s - residuals_training_steps

    # Signal forecasting
    Xtrain_signal = X[:s_tilde]
    ytrain_signal = y[:s_tilde]

    signal_forecaster = Forecaster(ESN_signal, num_features=52)
    signal_forecaster.fit(Xtrain_signal, ytrain_signal, warmup=10)

    warmup_X_signal = Xtrain_signal[-52:, :]
    denoised_signal_prediction = signal_forecaster.forecast(
        T=T - s_tilde, warmup_X=warmup_X_signal
    )

    # Compute residuals
    residuals = y[s_tilde:s].reshape(-1, 1) - denoised_signal_prediction[: s - s_tilde]
    residuals = residuals.flatten()

    # Residuals forecasting
    Xtrain_residuals, ytrain_residuals = create_training_data(residuals, 52)
    ytrain_residuals = ytrain_residuals.reshape(-1, 1)

    residuals_forecaster = Forecaster(ESN_residuals, num_features=52)
    residuals_forecaster.fit(Xtrain_residuals, ytrain_residuals, warmup=10)

    warmup_X_residuals = Xtrain_residuals[-52:, :]
    residuals_forecast = residuals_forecaster.forecast(
        T=T - s, warmup_X=warmup_X_residuals
    )

    # Compute lower bound
    lower_bound = denoised_signal_prediction[s - s_tilde : T] - k * residuals_forecast

    # Fault detection
    flag = False
    max_iter = T - s

    for i in range(0, max_iter, h):

        # Iteration parameters
        start_index = s + i
        end_index = start_index + h
        forecast_len = T - start_index

        # Iteration dataset
        curr_X = X[start_index:end_index]
        curr_y = y[start_index:end_index]

        # Train signal forecaster
        signal_forecaster.fit(curr_X, curr_y, warmup=0)

        # Predict denoised signal in generative mode
        curr_warmup_X = X[end_index - 52 : end_index]
        forecast = signal_forecaster.forecast(T=forecast_len, warmup_X=curr_warmup_X)

        # Flag condition
        for j in range(forecast_len - N + 1):
            flag = np.all(forecast[j : j + N] < lower_bound[i + j : i + j + N])
            if flag:
                return float(flag)

        return float(flag)
