import reservoirpy as rpy
import numpy as np


class Forecaster:

    model: rpy.model.Model
    num_features: int

    def __init__(self, esn: rpy.model.Model, num_features: int) -> None:
        self.num_features = num_features

        # Echo state network
        self.model = esn

    def fit(self, X: np.ndarray, y: np.ndarray, warmup: int = 10) -> None:
        if not self.model.fitted:
            self.model = self.model.fit(X, y, warmup=warmup)

    def forecast(self, T: int, warmup_X: np.ndarray) -> np.ndarray:

        assert warmup_X.shape[1] == self.num_features

        ypred = np.empty((T, 1))

        # Reset internal state and feed the last `memory` steps of time series
        warmup_y = self.model.run(warmup_X, reset=True)
        last_X = warmup_X[-1]

        # Generate first prediction
        x = np.concatenate((last_X[-(self.num_features - 1) :].flatten(), warmup_y[-1]))

        for i in range(T):
            prediction = self.model(x)
            x = np.concatenate((x[-(self.num_features - 1) :], prediction.flatten()))
            ypred[i] = prediction

        return ypred
