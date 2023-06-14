import reservoirpy
import numpy as np


class Forecaster():

    model: reservoirpy.model.Model
    X: np.ndarray
    y: np.ndarray

    def __init__(
        self, esn: reservoirpy.model.Model, X: np.ndarray, y: np.ndarray
    ) -> None:
        # Data
        self.X = X
        self.y = y

        # Echo state network
        self.model = esn

    def fit(self, warmup: int = 10) -> None:
        if not self.model.fitted:
            self.model = self.model.fit(self.X, self.y, warmup=warmup)

    def forecast(self, forecast_len: int, memory: int = 52) -> np.ndarray:

        # Reset internal state and feed as many steps as indicated by `memory`
        warmup_y = self.model.run(self.X[:-memory], reset=True)

        # Generate prediction
        ypred = np.empty((forecast_len, 1))
        x = warmup_y[-1].reshape(1, -1)

        for i in range(forecast_len):
            x = self.model(x)
            ypred[i] = x

        return ypred
