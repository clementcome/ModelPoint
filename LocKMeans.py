import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


class LocKMeans:
    """
    Implements a modified K-Means algorithm with size-constraints
    on the number of elements of each cluster
    This size constraint is enforced only during fitting

    Parameters:
    -----------
    n_clusters: int; number of clusters
    cluster_size: int, np.array; when int gives the maximal cluster size for every cluster
        when array (n_clusters,) gives a maximal cluster size for each cluster
    initial_centers: np.array; if None cluster centers are initialized randomly
        if array (n_clusters, num_features), it will be used as the initial centers for the algorithm
    """

    def __init__(self, n_clusters=8, cluster_size=None, max_iter=300, hide_pbar=False):
        self.n_clusters_ = n_clusters
        if type(cluster_size) == np.ndarray:
            self.cluster_size_ = cluster_size
        elif type(cluster_size) == int:
            self.cluster_size_ = np.repeat(cluster_size, n_clusters)
        else:
            self.cluster_size_ = None
        self.max_iter_ = max_iter
        self.cluster_centers_ = None
        self.hide_pbar_ = hide_pbar

    def fit(self, X, initial_centers=None):
        """
        Complexity is linear on cluster_size and second degree on n_clusters

        Parameters:
        -----------
        X: np.array (n_samples, n_features); data matrix
        initial_centers: np.array; if None cluster centers are initialized randomly
        if array (n_clusters, num_features), it will be used as the initial centers for the algorithm
        """
        n_samples, n_features = X.shape
        if initial_centers is None:
            center_indices = np.random.choice(n_samples, self.n_clusters_)
            initial_centers = X[center_indices]
        if self.cluster_size_ is None:
            avg_cluster_size = n_samples // self.n_clusters_ + 1
            self.cluster_size_ = np.repeat(avg_cluster_size, self.n_clusters_)
        self.cluster_centers_ = initial_centers
        # self.prediction_difference_ = np.zeros(self.max_iter_)
        self.labels_ = np.repeat(-1, X.shape[0])

        copy_X = X.copy()
        original_index = np.arange(n_samples)
        points_cluster_order = np.tile(
            np.arange(self.n_clusters_).reshape((1, -1)), (n_samples, 1)
        )

        for i in tqdm(range(self.max_iter_), disable=self.hide_pbar_):
            list_points_in_clusters = [[] for _ in range(self.n_clusters_)]
            list_cluster_size = [0 for _ in range(self.n_clusters_)]
            new_labels = np.repeat(-1, X.shape[0])
            dist_data_centers = cdist(copy_X, self.cluster_centers_)
            sort_index_data_centers = np.argsort(
                np.min(dist_data_centers, axis=1), kind="stable"
            )
            dist_data_centers = dist_data_centers[sort_index_data_centers]
            copy_X = copy_X[sort_index_data_centers]
            original_index = original_index[sort_index_data_centers]
            for idx in range(len(dist_data_centers)):
                visited_cluster = 0
                while visited_cluster < self.n_clusters_:
                    cluster_idx = np.argmin(dist_data_centers[idx])
                    if list_cluster_size[cluster_idx] < self.cluster_size_[cluster_idx]:
                        list_points_in_clusters[cluster_idx].append(copy_X[idx])
                        list_cluster_size[cluster_idx] += 1
                        new_labels[original_index[idx]] = cluster_idx
                        break
                    elif dist_data_centers[idx, cluster_idx] == np.inf:
                        break
                    else:
                        dist_data_centers[:, cluster_idx] = np.inf
                    visited_cluster += 1

            new_centers = np.zeros_like(self.cluster_centers_)
            for cluster_idx in range(self.n_clusters_):
                new_centers[cluster_idx] = np.mean(
                    np.array(list_points_in_clusters[cluster_idx]), axis=0
                )

            # If prediction are not changing between 2 iterations, we stop the algorithm
            if (new_labels == self.labels_).all():
                print(f"Convergence reached at step {i}")
                break

            self.labels_ = new_labels
            self.cluster_centers_ = new_centers

    def predict(self, X, centers=None):
        if centers is None:
            centers = self.cluster_centers_
        # list_cluster_size = [0 for _ in range(self.n_clusters_)]
        dist_data_centers = cdist(X, centers)
        # sort_index_data_centers = np.argsort(np.min(dist_data_centers, axis=1))
        labels = np.repeat(-1, X.shape[0])
        for i, arr_dist in enumerate(dist_data_centers):
            labels[i] = np.argmin(arr_dist)
        # for idx in sort_index_data_centers:
        #     visited_cluster = 0
        #     while visited_cluster < self.n_clusters_:
        #         cluster_idx = np.argmin(dist_data_centers[idx])
        #         if list_cluster_size[cluster_idx] < self.cluster_size_[cluster_idx]:
        #             labels[idx] = cluster_idx
        #             list_cluster_size[cluster_idx] += 1
        #             break
        #         elif dist_data_centers[idx, cluster_idx] == np.inf:
        #             break
        #         else:
        #             dist_data_centers[:, cluster_idx] = np.inf
        #         visited_cluster += 1
        return labels