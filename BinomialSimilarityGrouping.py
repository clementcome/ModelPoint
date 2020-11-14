import numpy as np
import scipy.stats as stats
from scipy.special import binom
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, single, ward
import pickle


class BSG:
    """
    Performs similarity grouping on data that is spatially distributed
    and has also a Binomial distributed value associated to it.
    """

    def __init__(self, n_estimation=100, p_penalization=None, verbose=False):
        self.n_estimation_ = n_estimation
        self.linkage_matrix_ = None
        if p_penalization is None:
            self.p_penalization_ = lambda x: 1 + x
        else:
            self.p_penalization_ = p_penalization
        self.verbose_ = verbose

    def save(self, file):
        with open(file, "wb") as f:
            pickle.dump(
                {
                    "n_estimation": self.n_estimation_,
                    "linkage_matrix": self.linkage_matrix_,
                    "labels": self.labels_,
                },
                f,
            )

    def load(self, file):
        with open(file, "rb") as f:
            attributes = pickle.load(f)
        self.n_estimation_ = attributes["n_estimation"]
        self.linkage_matrix_ = attributes["linkage_matrix"]
        self.labels_ = attributes["labels"]

    def dist_proba(self, xp, yp):
        x_min, x_max = int(min(xp[0], yp[0])), int(max(xp[0], yp[0]))
        return (
            np.sum(self.proba_array_dict_[xp[1]][x_min:x_max])
            + np.sum(self.proba_array_dict_[yp[1]][x_min:x_max])
        ) / 2

    def fit_hierarchy(self, X, y, p):
        n_sample, n_feature = X.shape
        y = y.reshape(-1, 1)
        if type(p) == float:
            p = np.repeat(p, n_sample)
        self.p_ = p.reshape(-1, 1)

        unique_p_arr = np.unique(self.p_)

        s = np.arange(self.n_estimation_ + 1)
        self.proba_array_dict_ = {
            unique_p: stats.binom.pmf(s, self.n_estimation_, unique_p)
            for unique_p in unique_p_arr
        }
        if self.verbose_:
            print("Compute spatial distance")
        dist = pdist(X)
        dist_p = pdist(self.p_)

        if self.verbose_:
            print("Compute probability distance")
        yp = np.concatenate((y, self.p_), axis=1)
        dist_distribution = pdist(yp, metric=self.dist_proba)

        combined_dist = dist / (1 - dist_distribution) * self.p_penalization_(dist_p)

        if self.verbose_:
            print("Compute Linkage matrix")
        self.linkage_matrix_ = ward(combined_dist)
        return self.linkage_matrix_

    def predict_hierarchy(self, max_cluster=10):
        self.labels_ = fcluster(
            self.linkage_matrix_, t=max_cluster, criterion="maxclust"
        )
        return self.labels_


# np.random.seed(42)
# n_estimation = 100
# n_cluster = 100
# p_list = [0.3, 0.5, 0.8]
# p_arr = np.random.choice(p_list, size=n_cluster)
# cluster_centers = np.random.rand(n_cluster, 2)
# binom_sample = np.array([stats.binom.rvs(n_estimation, p) for p in p_arr])
# bsg = BSG(n_estimation)
# linkage_matrix = bsg.fit_hierarchy(cluster_centers, binom_sample, p_arr)
# labels = bsg.predict_hierarchy(15)
