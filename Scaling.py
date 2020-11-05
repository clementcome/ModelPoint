import numpy as np


class LogMinMaxScaler:
    def __init__(self):
        self.x_min_ = None
        self.x_max_ = None

    def fit(self, X):
        self.x_min_ = np.min(X, axis=0)
        self.x_max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        return (np.log(X) - np.log(self.x_min_)) / (
            np.log(self.x_max_) - np.log(self.x_min_)
        )

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ExpMinMaxScaler:
    def __init__(self):
        self.x_min_ = None
        self.x_max_ = None

    def fit(self, X):
        self.x_min_ = np.min(X, axis=0)
        self.x_max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        return (np.exp(-X) - np.exp(-self.x_min_)) / (
            np.exp(-self.x_max_) - np.exp(-self.x_min_)
        )

    def fit_transform(self, X):
        return self.fit(X).transform(X)