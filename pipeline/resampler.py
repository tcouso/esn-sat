import pandas as pd

from .pipeline import Pipeline

class Resampler(Pipeline):

  def __init__(self, rule="w") -> None:
    super().__init__()
    self.rule = rule

  def map(self, ts: pd.Series) -> pd.Series:
    # Resample by period
    resampled_ts = ts.resample(self.rule).median()

    # Fill missing values by interpolation
    resampled_ts = resampled_ts.interpolate()

    # Fill initial missing values by backward fill
    resampled_ts = resampled_ts.bfill()

    return resampled_ts
