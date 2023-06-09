import statsmodels
import pandas as pd

from .pipeline import Pipeline


class SmoothingFilter(Pipeline):

  seasonal_periods: int
  model: statsmodels.tsa.base.tsa_model.TimeSeriesModel

  def __init__(self, seasonal_periods:int, model: statsmodels.tsa.base.tsa_model.TimeSeriesModel) -> None:
    super().__init__()
    self.seasonal_periods = seasonal_periods
    self.model = model

  def map(self, ts: pd.Series) -> pd.Series:

    # Instance and fit model
    fit = self.model(ts, trend="add", seasonal="add", seasonal_periods=self.seasonal_periods).fit()

    # Compute filtered values
    hw_smoothed_ts = fit.fittedvalues

    return hw_smoothed_ts

