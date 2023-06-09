import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def downsample_ts(ts: pd.Series) -> pd.Series:
  """
  Description: 
  This function takes a pandas Series `ts` as input and 
  downsamples it by week using the median of the data within each week. 

  Parameters:
  - `ts` (pd.Series): The input pandas Series containing the time series data.

  Returns:
  - pd.Series: The resampled time series.

  """
  # Downsample by week
  resampled_ts = ts.resample("W").median()

  # Fill missing values by interpolation
  resampled_ts = resampled_ts.interpolate()

  # Fill initial missing values by backward fill
  resampled_ts = resampled_ts.bfill()

  return resampled_ts


def moving_std_filter(ts: pd.Series, window_size:int=26, std_range:int=4) -> pd.Series:
  """
  Description: 
  This function takes a pandas Series `ts` as input and filters 
  the data based on the rolling mean and standard deviation 
  within a specified window.

  Parameters:
  - `ts` (pd.Series): The input pandas Series containing the time series data.
  - `window_size` (int): The size of the window used to compute the rolling mean and standard deviation (default: 26).
  - `std_range` (int): The number of standard deviations used to calculate the upper and lower bounds (default: 4).

  Returns:
  - pd.Series: The filtered time series.

  """
  # Compute rolling metrics
  rolling_mean = ts.rolling(window=window_size).mean()
  rolling_std = ts.rolling(window=window_size).std()

  # Compute thresholds
  rolling_threshold_ub = rolling_mean + (std_range / 2) * rolling_std
  rolling_threshold_lb = rolling_mean - (std_range / 2) * rolling_std

  # Filter out of bounds values
  filtered_ts = ts[(ts < rolling_threshold_ub) & (ts > rolling_threshold_lb)]

  # Interpolate missing values
  filtered_ts = filtered_ts.resample("W").interpolate()

  # Fill initial missing values by backward fill
  filtered_ts = filtered_ts.bfill()



  return filtered_ts


def holt_winters_filter(ts:pd.Series, period:int=52) -> pd.Series:
  """
  Description: 
  This function takes a pandas Series `ts` as input and smooths it 
  using the Holt-Winters exponential smoothing method.

  Parameters:
  - `ts` (pd.Series): The input pandas Series containing the time series data.
  - `period` (int): The number of periods per season for the seasonal component (default: 52).

  Returns:
  - pd.Series: The smoothed time series using the Holt-Winters method.
  """

  model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=period)
  fit = model.fit()
  hw_smoothed_ts = fit.fittedvalues

  return hw_smoothed_ts