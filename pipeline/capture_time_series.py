import pandas as pd

from .pipeline import Pipeline

class CaptureTimeSeries(Pipeline):
  """Pipeline task to capture time series"""

  src: pd.Series

  def __init__(self, time_series: pd.Series):
    super().__init__()
    self.src = time_series

  def generator(self):

    data = self.src

    yield self.map(data)
