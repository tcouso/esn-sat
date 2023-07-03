import reservoirpy
import numpy as np


class Forecaster:

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

    def forecast(
        self, model: reservoirpy.model.Model, T: int, memory: int = 52
    ) -> np.ndarray:

        ypred = np.empty((T, 1))

        # Reset internal state and feed the last 52 steps of time series
        warmup_y = model.run(self.X[-memory:], reset=True)

        # Generate first prediction
        x = np.concatenate((self.y[-(memory - 1) :].flatten(), warmup_y[-1]))

        for i in range(T):
            prediction = model.run(x)
            x = np.concatenate((x[-(memory - 1) :], prediction.flatten()))
            ypred[i] = prediction

        return ypred
