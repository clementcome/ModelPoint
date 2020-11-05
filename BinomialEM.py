import numpy as np
from scipy.special import binom
from tqdm import tqdm
import plotly.graph_objects as go
import scipy.stats as stats


def plot_binomial_mixture(p_values, proportions=None, S=None, n_estimation=100):
    if proportions is None:
        proportions = np.ones_like(p_values)
    fig = go.Figure()
    for p, prop in zip(p_values, proportions):
        x = np.arange(
            stats.binom.ppf(0.01, n_estimation, p),
            stats.binom.ppf(0.99, n_estimation, p),
        )
        fig.add_scatter(
            x=x,
            y=prop * stats.binom.pmf(x, n_estimation, p),
            mode="lines",
            name=f"p={p:.3f}",
        )
    if S is not None:
        fig.add_histogram(x=S, nbinsx=30, histnorm="probability density")
    fig.show()


class BinomialEM:
    """
    Performing Expectation Maximization algorithm on Binomial data
    """

    def __init__(
        self,
        n_components: int = 1,
        n_estimation: int = 100,
        max_iter: int = 100,
        hide_pbar: bool = False,
    ):
        self.n_components_ = n_components
        self.n_estimation_ = n_estimation
        self.max_iter_ = max_iter
        self.hide_pbar_ = hide_pbar
        self.lambd_ = None
        self.p_ = None

    def f(self, i: int):
        return (
            binom(self.n_estimation_, i)
            * self.p_ ** i
            * (1 - self.p_) ** (self.n_estimation_ - i)
        )

    def fit(self, S: np.ndarray, initial_p: np.ndarray = None):

        self.lambd_ = np.repeat(1 / self.n_components_, self.n_components_)
        if initial_p is not None:
            self.p_ = initial_p
        else:
            min_p = np.min(S / self.n_estimation_)
            max_p = np.max(S / self.n_estimation_)
            self.p_ = np.linspace(min_p, max_p, self.n_components_)
        S = S.reshape((-1, 1))
        n_sample = S.shape[0]

        for t in tqdm(range(self.max_iter_), disable=self.hide_pbar_):
            # E-step
            self.prob_ = np.apply_along_axis(self.f, 1, S)
            P = self.lambd_ * self.prob_
            P = P / np.sum(P, axis=1).reshape(-1, 1)

            # M-step
            self.lambd_ = np.sum(P, axis=0) / n_sample
            self.p_ = np.sum(P * S, axis=0) / (
                n_sample * self.lambd_ * self.n_estimation_
            )
        self.labels_ = np.apply_along_axis(np.argmax, 1, P)

    def predict(self, S: np.ndarray):
        S = S.reshape((-1, 1))
        P = self.lambd_ * np.apply_along_axis(self.f, 1, S)
        P = P / np.sum(P, axis=1).reshape(-1, 1)
        labels = np.apply_along_axis(np.argmax, 1, P)
        return labels
