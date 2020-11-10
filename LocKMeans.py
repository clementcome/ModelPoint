import numpy as np
from numpy.core.defchararray import replace
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
        compute_loss=False,
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
        self.compute_loss_ = compute_loss
        if self.compute_loss_:
            self.loss_history_ = np.zeros(max_iter)

    def compute_std(self, labels=None, remove_outlier=True):
        if labels is None:
            labels = self.labels_
        if remove_outlier:
            true_labels = labels[np.where(labels > -1)]
        else:
            true_labels = labels.copy()
        unique_labels, counts = np.unique(true_labels, return_counts=True)
        n_labels = unique_labels.shape[0]
        if remove_outlier:
            total_counts = np.zeros(self.n_clusters_)
        else:
            total_counts = np.zeros(self.n_clusters_ + 1)

        total_counts[:n_labels] = counts
        return np.std(total_counts)

    def initial_centers(self, X: np.ndarray, init_mode: str) -> np.ndarray:
        """
        Initialize centers for fitting LocKMeans according to 
        the mode of initialization

        Parameters
        ----------
        X : np.ndarray
            Array to fit
        init_mode : str
            Mode of initialization

        Returns
        -------
        np.ndarray
            Initial centers
        """
        n_samples = X.shape[0]
        if init_mode == "kmeans":
            print("Initialization with KMeans")
            verbose = 0 if self.hide_pbar_ else 1
            self.km_ = KMeans(
                self.n_clusters_, max_iter=self.max_iter_ // 2, verbose=verbose
            )
            self.km_.fit(X)
            initial_centers = self.km_.cluster_centers_
            print("Initialization finished")
        else:
            center_indices = np.random.choice(n_samples, self.n_clusters_)
            initial_centers = X[center_indices]
        return initial_centers

    def total_update_loop(self, copy_X: np.ndarray, num_threads: int):
        n_samples = copy_X.shape[0]
        visited_cluster_through_iteration = np.zeros(n_samples)

        # Definition of the similarity search component
        index_search = nmslib.init(space="l2")
        index_search.addDataPointBatch(self.cluster_centers_.astype(np.float32))
        index_search.createIndex()

        # Initialize with empty cluster composition
        list_points_in_clusters = [[] for _ in range(self.n_clusters_)]
        list_cluster_size = np.zeros(self.n_clusters_)
        new_labels = np.repeat(-1, copy_X.shape[0])

        # Retrieve the k-nearest cluster for every sample
        knn_result = index_search.knnQueryBatch(
            copy_X, k=self.truncate_cluster_, num_threads=num_threads,
        )
        idx_data_centers, dist_data_centers = list(zip(*knn_result))
        idx_data_centers = np.array(idx_data_centers)
        dist_data_centers = np.array(dist_data_centers)

        # Retrieve the ordered index of the samples by their minimum distance to a cluster
        sort_index_data_centers = np.argsort(np.min(dist_data_centers, axis=1))

        # Loop over the points by the order defined above
        for point_index in sort_index_data_centers:
            # Search a cluster to fit in (ie its the closest
            # and its maximal size has not been reached)
            visited_cluster = 0
            while visited_cluster < self.truncate_cluster_:
                cluster_idx = idx_data_centers[point_index, visited_cluster]
                visited_cluster += 1
                if list_cluster_size[cluster_idx] < self.cluster_size_[cluster_idx]:
                    list_points_in_clusters[cluster_idx].append(copy_X[point_index])
                    list_cluster_size[cluster_idx] += 1
                    new_labels[point_index] = cluster_idx
                    break

            # For monitoring purposes, remember the number of cluster visited to find a fit
            visited_cluster_through_iteration[point_index] = visited_cluster

        # Compute the new centers
        new_centers = self.cluster_centers_.copy()
        for cluster_idx in range(self.n_clusters_):
            if list_cluster_size[cluster_idx] > 0:
                new_centers[cluster_idx] = np.mean(
                    np.array(list_points_in_clusters[cluster_idx]), axis=0
                )

        return new_centers, new_labels, visited_cluster_through_iteration

    def batch_update_loop(
        self, copy_X: np.ndarray, batch_size: int, alpha: float, num_threads: int
    ):
        # Initialize batch parameters
        n_samples = copy_X.shape[0]
        n_batch = n_samples // batch_size
        batch_cluster_size = self.cluster_size_ // n_batch + 1
        visited_cluster_through_iteration = np.zeros(n_samples)
        index_batch_split = np.split(
            np.random.choice(n_samples, n_batch * batch_size, replace=False), n_batch
        )
        cluster_centers = self.cluster_centers_.copy().astype(np.float32)

        new_labels = np.repeat(-1, copy_X.shape[0])

        # Loop over batches
        for index_batch in index_batch_split:
            # Define the subset of the batch
            X_batch = copy_X[index_batch]

            # Definition of the similarity search component
            index_search = nmslib.init(space="l2")
            index_search.addDataPointBatch(cluster_centers)
            index_search.createIndex()

            # Initialize with empty cluster composition
            list_points_in_clusters = [[] for _ in range(self.n_clusters_)]
            list_cluster_size = np.zeros(self.n_clusters_)

            # Retrieve the k-nearest cluster for every sample in the batch
            knn_result = index_search.knnQueryBatch(
                X_batch, k=self.truncate_cluster_, num_threads=num_threads,
            )
            idx_data_centers, dist_data_centers = list(zip(*knn_result))
            idx_data_centers = np.array(idx_data_centers)
            dist_data_centers = np.array(dist_data_centers)

            # Retrieve the ordered index of the samples by their minimum distance to a cluster
            sort_index_data_centers = np.argsort(np.min(dist_data_centers, axis=1))

            # Loop over the points by the order defined above
            for idx in sort_index_data_centers:
                point_index = index_batch[idx]
                # Search a cluster to fit in (ie its the closest
                # and its maximal size has not been reached)
                visited_cluster = 0
                while visited_cluster < self.truncate_cluster_:
                    cluster_idx = idx_data_centers[idx, visited_cluster]
                    visited_cluster += 1
                    if list_cluster_size[cluster_idx] < batch_cluster_size[cluster_idx]:
                        list_points_in_clusters[cluster_idx].append(copy_X[point_index])
                        list_cluster_size[cluster_idx] += 1
                        new_labels[point_index] = cluster_idx
                        break

                # For monitoring purposes, remember the number of cluster visited to find a fit
                visited_cluster_through_iteration[point_index] = visited_cluster

            # Compute the new centers
            new_centers = cluster_centers.copy()
            for cluster_idx in range(self.n_clusters_):
                if list_cluster_size[cluster_idx] > 0:
                    new_centers[cluster_idx] = np.mean(
                        np.array(list_points_in_clusters[cluster_idx]), axis=0
                    )
            cluster_centers = (1 - alpha) * cluster_centers + alpha * new_centers

        return cluster_centers, new_labels, visited_cluster_through_iteration

    def fit(
        self,
        X,
        init_mode="random",
        fit_mode="total",
        batch_size=100,
        alpha=0.5,
        num_threads=4,
        print_std=False,
    ):
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
        self.cluster_centers_ = self.initial_centers(X, init_mode)
        if self.cluster_size_ is None:
            avg_cluster_size = n_samples // self.n_clusters_ + 1
            self.cluster_size_ = np.repeat(avg_cluster_size, self.n_clusters_)
        self.visited_cluster_through_iteration_list_ = np.zeros(
            (self.max_iter_, n_samples)
        )
        self.labels_ = np.repeat(-1, X.shape[0])

        copy_X = X.copy().astype(np.float32)

        max_iter = self.max_iter_
        if init_mode == "kmeans":
            max_iter = max_iter // 2
            self.visited_cluster_through_iteration_list_ = np.zeros(
                (self.max_iter_ // 2, n_samples)
            )

        for i in tqdm(range(max_iter), disable=self.hide_pbar_):
            if fit_mode == "batch":
                (
                    new_centers,
                    new_labels,
                    visited_cluster_through_iteration,
                ) = self.batch_update_loop(copy_X, batch_size, alpha, num_threads)
            else:
                (
                    new_centers,
                    new_labels,
                    visited_cluster_through_iteration,
                ) = self.total_update_loop(copy_X, num_threads)

            self.visited_cluster_through_iteration_list_[
                i
            ] = visited_cluster_through_iteration

            if self.compute_loss_ or print_std:
                std = self.compute_std(new_labels, remove_outlier=False)
                if self.compute_loss_:
                    self.loss_history_[i] = std
                if print_std:
                    print(std)

            # If prediction are not changing between 2 iterations, we stop the algorithm
            if (new_labels == self.labels_).all():
                print(f"Convergence reached at step {i}")
                break

            self.labels_ = new_labels
            self.cluster_centers_ = new_centers
        self.index_search_ = nmslib.init(space="l2")
        self.index_search_.addDataPointBatch(self.cluster_centers_.astype(np.float32))
        self.index_search_.createIndex()

    def predict(self, X, limit_cluster_size=False):
        knn_result = self.index_search_.knnQueryBatch(X, k=1, num_threads=4)
        idx_data_centers, dist_data_centers = list(zip(*knn_result))
        idx_data_centers = np.array(idx_data_centers)
        dist_data_centers = np.array(dist_data_centers)
        list_cluster_size = np.zeros(self.n_clusters_)
        if limit_cluster_size:
            labels = np.repeat(-1, X.shape[0])
            order = np.argsort(np.min(dist_data_centers, axis=1))
            for point_index in tqdm(order):
                cluster_idx = idx_data_centers[point_index, 0]
                if list_cluster_size[cluster_idx] < self.cluster_size_[cluster_idx]:
                    list_cluster_size[cluster_idx] += 1
                    labels[point_index] = cluster_idx
                else:
                    idx_nn, _ = self.index_search_.knnQuery(
                        X[point_index], k=self.n_clusters_
                    )
                    # idx_nn, _ = list(zip(*knn_point))
                    for cluster_idx in idx_nn:
                        if (
                            list_cluster_size[cluster_idx]
                            < self.cluster_size_[cluster_idx]
                        ):
                            list_cluster_size[cluster_idx] += 1
                            labels[point_index] = cluster_idx
                            break
            return labels
        else:
            idx_data_centers = np.array(idx_data_centers).reshape(-1)
            return idx_data_centers


# np.random.seed(42)
# n_cluster = 100
# n_estimation = 100
# X = np.random.randn(n_cluster * n_estimation, 2).astype(np.float32)

# lkm = LocKMeans(n_clusters=n_cluster, cluster_size=n_estimation, max_iter=100)
# lkm.fit(X, init_mode="random")
# lkm.predict(X, limit_cluster_size=True)
