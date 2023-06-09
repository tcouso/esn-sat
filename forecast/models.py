import reservoirpy as rpy
import numpy as np

from reservoirpy.nodes import Reservoir, Ridge

# ESN definition

# Layers
reservoir = Reservoir(1000, lr=.2, sr=.9, seed=0)
readout = Ridge(ridge=1e-7)

# Feedback connection
reservoir <<= readout

# Echo state network
ESN = reservoir >> readout

def forecast(model: rpy.model.Model, 
                      forecast_len:int,
                      memory: int,
                    x: float,
                      warmup:int=10,
                      ) -> np.ndarray:

  # Generate prediction
  ypred = np.empty((forecast_len, 1))

  for i in range(forecast_len):
    x = model(x)
    ypred[i] = x
  
  return ypred

