import pandas as pd

from .pipeline import Pipeline

class RollingFilter(Pipeline):

  window_size: int
  std_range: int

  def __init__(self, window_size:int=26, std_range:int=4) -> None:
    super().__init__()
    self.window_size = window_size
    self.std_range = std_range

  def map(self, ts:pd.Series)-> pd.Series:
    # Compute rolling metrics
    rolling_mean = ts.rolling(window=self.window_size).mean()
    rolling_std = ts.rolling(window=self.window_size).std()

    # Compute thresholds
    rolling_threshold_ub = rolling_mean + (self.std_range / 2) * rolling_std
    rolling_threshold_lb = rolling_mean - (self.std_range / 2) * rolling_std

    # Filter out of bounds values
    filtered_ts = ts[(ts < rolling_threshold_ub) & (ts > rolling_threshold_lb)]

    # Interpolate missing values
    filtered_ts = filtered_ts.resample("W").interpolate()

    # Fill initial missing values by backward fill
    filtered_ts = filtered_ts.bfill()
    
    return filtered_ts