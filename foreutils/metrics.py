import numpy as np
import pandas as pd


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Description:
    Computer the Mean Absolute Percentage Error (MAPE) between
    two vectors

    Parameters:

    Returns:

    """
    # Compute the absolute percentage error for each value
    ape = np.abs((y_true - y_pred) / y_true) * 100

    # Calculate the mean of the absolute percentage errors
    mape = np.mean(ape)

    return mape


def SNR(ts: pd.Series, window_size: int = 26) -> float:
    # Extract signal and noise
    smoothed_ts = ts.rolling(window=window_size).median()
    noise = ts - smoothed_ts

    # Compute powers
    signal_pow = np.var(smoothed_ts)
    noise_pow = np.var(noise)

    # Compute snr
    snr = 10 * np.log10(signal_pow / noise_pow)

    return snr
