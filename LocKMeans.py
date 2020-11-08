import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.cluster import KMeans
import nmslib


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

    def __init__(
        self,
        n_clusters=8,
        cluster_size=None,
        truncate_cluster=2,
        max_iter=300,
        hide_pbar=False,
    ):
        self.n_clusters_ = n_clusters
        if type(cluster_size) == np.ndarray:
            self.cluster_size_ = cluster_size
        elif type(cluster_size) == int:
            self.cluster_size_ = np.repeat(cluster_size, n_clusters)
        else:
            self.cluster_size_ = None
        self.truncate_cluster_ = truncate_cluster
        self.max_iter_ = max_iter
        self.cluster_centers_ = None
        self.hide_pbar_ = hide_pbar

    def fit(self, X, init_mode="random"):
        """
        Complexity is linear on cluster_size and second degree on n_clusters

        Parameters:
        -----------
        X: np.array (n_samples, n_features); data matrix
        init_mode: str default "random"
            if "random": cluster initialization is random within the points
            if "kmeans": cluster initialization is done through kmeans algorithm
                with iteration equal to self.max_iter_ // 2 and the LocKMeans will
                iterate for the other self.max_iter_ // 2
        """
        n_samples, n_features = X.shape
        if init_mode == "kmeans":
            print("Initialization with KMeans")
            self.km_ = KMeans(self.n_clusters_, max_iter=self.max_iter_ // 2)
            self.km_.fit(X)
            initial_centers = self.km_.cluster_centers_
            print("Initialization finished")
        else:
            center_indices = np.random.choice(n_samples, self.n_clusters_)
            initial_centers = X[center_indices]
        if self.cluster_size_ is None:
            avg_cluster_size = n_samples // self.n_clusters_ + 1
            self.cluster_size_ = np.repeat(avg_cluster_size, self.n_clusters_)
        self.cluster_centers_ = initial_centers
        self.visited_cluster_through_iterations_ = np.zeros((self.max_iter_, n_samples))
        self.labels_ = np.repeat(-1, X.shape[0])

        copy_X = X.copy()
        original_index = np.arange(n_samples)
        points_cluster_order = np.tile(
            np.arange(self.n_clusters_).reshape((1, -1)), (n_samples, 1)
        )

        max_iter = self.max_iter_
        if init_mode == "kmeans":
            max_iter = max_iter // 2
            self.visited_cluster_through_iterations_ = np.zeros(
                (self.max_iter_ // 2, n_samples)
            )

        for i in tqdm(range(max_iter), disable=self.hide_pbar_):
            # Definition of the similarity search component
            index_search = nmslib.init(space="l2")
            index_search.addDataPointBatch(self.cluster_centers_.astype(np.float32))
            index_search.createIndex()

            list_points_in_clusters = [[] for _ in range(self.n_clusters_)]
            list_cluster_size = [0 for _ in range(self.n_clusters_)]
            new_labels = np.repeat(-1, X.shape[0])
            knn_result = index_search.knnQueryBatch(
                copy_X.astype(np.float32), k=self.truncate_cluster_, num_threads=4
            )
            idx_data_centers, dist_data_centers = list(zip(*knn_result))
            idx_data_centers = np.array(idx_data_centers)
            dist_data_centers = np.array(dist_data_centers)
            sort_index_data_centers = np.argsort(
                np.min(dist_data_centers, axis=1), kind="stable"
            )
            idx_data_centers = idx_data_centers[sort_index_data_centers]
            dist_data_centers = dist_data_centers[sort_index_data_centers]
            copy_X = copy_X[sort_index_data_centers]
            original_index = original_index[sort_index_data_centers]
            for idx in range(len(dist_data_centers)):
                visited_cluster = 0
                while visited_cluster < self.truncate_cluster_:
                    cluster_idx = idx_data_centers[idx, visited_cluster]
                    visited_cluster += 1
                    if list_cluster_size[cluster_idx] < self.cluster_size_[cluster_idx]:
                        list_points_in_clusters[cluster_idx].append(copy_X[idx])
                        list_cluster_size[cluster_idx] += 1
                        new_labels[original_index[idx]] = cluster_idx
                        break
                self.visited_cluster_through_iterations_[
                    i, original_index[idx]
                ] = visited_cluster

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
            self.index_search_ = nmslib.init(space="l2")
            self.index_search_.addDataPointBatch(self.cluster_centers_)
            self.index_search_.createIndex()

    def predict(self, X):
        knn_result = self.index_search_.knnQueryBatch(X, k=1, num_threads=4)
        idx_data_centers, dist_data_centers = list(zip(*knn_result))
        idx_data_centers = np.array(idx_data_centers).reshape(-1)
        # dist_data_centers = np.array(dist_data_centers)
        # labels = np.repeat(-1, X.shape[0])
        # for i, arr_dist in enumerate(dist_data_centers):
        #     labels[i] = np.argmin(arr_dist)
        return idx_data_centers


# np.random.seed(42)
# n_cluster = 100
# n_estimation = 100
# X = np.random.randn(n_cluster * n_estimation, 2).astype(np.float32)

# lkm = LocKMeans(n_clusters=n_cluster, cluster_size=n_estimation, max_iter=100)
# lkm.fit(X, init_mode="random")
