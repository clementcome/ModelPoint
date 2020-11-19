from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler

from Scaling import LogMinMaxScaler

from LocKMeans import LocKMeans

from BinomialEM import BinomialEM

from BinomialSimilarityGrouping import BSG


class ModelPointDefiner:
    def __init__(
        self, data: pd.DataFrame = None, target: str = None, hide_pbar: bool = False
    ) -> None:
        if (data is None) or (target is None):
            print(
                "You did not provide a dataset, this behavior suppose that you will load a previously saved ModelPointDefiner before using it"
            )
        else:
            self.data_ = data
            self.N_ = self.data_.shape[0]
            self.variables_ = data.columns.drop(target)
            self.target_ = target
            self.scaler_n_ = MinMaxScaler()
            self.data_n_ = self.scaler_n_.fit_transform(
                self.data_[self.variables_].values
            )
            self.epsilon_ = 1e-6
            self.scaler_ln_ = LogMinMaxScaler()
            self.data_ln_ = self.scaler_ln_.fit_transform(self.data_n_ + self.epsilon_)
            self.n_model_ = 10
        self.hide_pbar_ = hide_pbar
        self.cluster_labels_ = None
        self.n_estimation_ = None
        self.points_in_cluster_ = None
        self.binomial_data_ = None
        self.p_labels_ = None
        self.p_values_ = None
        self.n_model_ = None
        self.hierarchy_ = None
        self.lkm_ = LocKMeans()
        self.bem_ = BinomialEM()
        self.bsg_ = BSG()

    def save(self, filename):
        with open(f"{filename}_mpd_attributes.pkl", "wb") as f:
            pickle.dump(
                {
                    "data": self.data_,
                    "target": self.target_,
                    "cluster_labels": self.cluster_labels_,
                    "n_estimation": self.n_estimation_,
                    "points_in_cluster": self.points_in_cluster_,
                    "binomial_data": self.binomial_data_,
                    "p_labels": self.p_labels_,
                    "p_values": self.p_values_,
                    "n_model": self.n_model_,
                    "hierarchy": self.hierarchy_,
                    "hide_pbar": self.hide_pbar_,
                },
                f,
            )
        self.lkm_.save(f"{filename}_mpd_lkm.pkl")
        self.bem_.save(f"{filename}_mpd_bem.pkl")
        self.bsg_.save(f"{filename}_mpd_bsg.pkl")

    def load(self, filename):
        with open(f"{filename}_mpd_attributes.pkl", "rb") as f:
            attributes = pickle.load(f)
        self.__init__(attributes["data"], attributes["target"], attributes["hide_pbar"])
        self.cluster_labels_ = attributes["cluster_labels"]
        self.n_estimation_ = attributes["n_estimation"]
        self.points_in_cluster_ = attributes["points_in_cluster"]
        self.binomial_data_ = attributes["binomial_data"]
        self.p_labels_ = attributes["p_labels"]
        self.p_values_ = attributes["p_values"]
        self.n_model_ = attributes["n_model"]
        self.hierarchy_ = attributes["hierarchy"]
        self.lkm_.load(f"{filename}_mpd_lkm.pkl")
        self.bem_.load(f"{filename}_mpd_bem.pkl")
        self.bsg_.load(f"{filename}_mpd_bsg.pkl")

    def fit_cluster(
        self,
        n_estimation: int,
        init_mode: str = "random",
        fit_mode: str = "total",
        batch_size: int = 10000,
        alpha: float = 0.3,
        truncate_cluster: int = 20,
        max_iter: int = 100,
        compute_loss: bool = False,
        num_threads: int = 4,
        print_std: bool = False,
    ) -> None:
        n_clusters = self.N_ // n_estimation
        cluster_size = n_estimation
        self.n_estimation_ = n_estimation
        self.lkm_ = LocKMeans(
            n_clusters,
            cluster_size,
            truncate_cluster,
            max_iter,
            self.hide_pbar_,
            compute_loss=compute_loss,
        )
        self.lkm_.fit(
            self.data_[self.variables_].values,
            init_mode=init_mode,
            fit_mode=fit_mode,
            batch_size=batch_size,
            alpha=alpha,
            num_threads=num_threads,
            print_std=print_std,
        )
        self.cluster_labels_ = self.lkm_.predict(
            self.data_[self.variables_].values, limit_cluster_size=True
        )

    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        return self.lkm_.predict(X)

    def fit_EM(self, p: int = None, max_iter: int = 100) -> None:
        def aic(model: BinomialEM):
            return (
                np.sum(np.log(np.sum(model.prob_ * model.lambd_, axis=1)))
                - model.n_components_ * 2
            )

        self.points_in_cluster_ = self.cluster_labels_ > -1
        self.binomial_data_ = (
            self.data_[self.points_in_cluster_]
            .groupby(self.cluster_labels_[self.points_in_cluster_])[self.target_]
            .sum()
            .values
        )
        if p is None:
            p = 1
            max_aic = -np.inf
            max_reached = False
            print("Performing model selection with AIC")
            while not (max_reached):
                print("Current p is", p)
                bem = BinomialEM(p, self.n_estimation_, hide_pbar=True)
                bem.fit(self.binomial_data_)
                current_aic = aic(bem)
                if current_aic < max_aic:
                    max_reached = True
                else:
                    max_aic = current_aic
                    p += 1
        self.bem_ = BinomialEM(p, self.n_estimation_, max_iter, self.hide_pbar_)
        self.bem_.fit(self.binomial_data_)
        self.p_labels_ = self.bem_.labels_
        self.p_values_ = self.bem_.p_[self.p_labels_]

    def p_of_cluster(self, cluster_label: int) -> float:
        # p_label = self.p_labels_[cluster_label]
        p = self.p_labels_[cluster_label]
        return p

    def fit_BSG(self, columns=None, normalization=None):
        data = []
        for idx, column in enumerate(columns):
            i = list(self.variables_).index(column)
            if normalization is None:
                data.append(self.data_.values[:, i].reshape(-1, 1))
            elif normalization[idx] == "n":
                data.append(self.data_n_[:, i].reshape(-1, 1))
            elif normalization[idx] == "ln":
                data.append(self.data_ln_[:, i].reshape(-1, 1))
            else:
                data.append(self.data_.values[:, i].reshape(-1, 1))
        if len(data) > 1:
            data = np.concatenate(data, axis=1)
        else:
            data = data[0]
        bsg_data = (
            pd.DataFrame(data[self.points_in_cluster_])
            .groupby(self.cluster_labels_[self.points_in_cluster_])
            .mean()
            .values
        )
        self.bsg_ = BSG(self.n_estimation_)
        self.bsg_.fit_hierarchy(bsg_data, self.binomial_data_, self.p_values_)

    def set_hierarchy(self, n_model):
        self.n_model_ = n_model
        self.hierarchy_ = self.bsg_.predict_hierarchy(self.n_model_)

    def model_from_cluster(self, cluster_label):
        if self.hierarchy_ is None:
            self.set_hierarchy(self.n_model_)
        return self.hierarchy_[cluster_label]

    def model_from_data(self, X: np.ndarray):
        cluster_labels = self.predict_cluster(X)
        model_point_labels = np.apply_along_axis(
            self.model_from_cluster, 0, cluster_labels
        )
        return model_point_labels


def threshold_1d(
    definer: ModelPointDefiner,
    dimension: str,
    step_number: int = 10000,
    q: float = 0.9,
    return_fake_data: bool = False,
):
    n_model = definer.n_model_
    data = pd.DataFrame(
        np.unique(
            np.quantile(definer.data_[dimension], np.linspace(0, 1, step_number))
        ),
        columns=[dimension],
    )
    cst_variables = definer.variables_.drop(dimension)
    dimension_bin = pd.cut(
        definer.data_[dimension],
        np.unique(
            np.quantile(definer.data_[dimension], np.linspace(0, 1, step_number))
        ),
        include_lowest=True,
    )
    group_data = definer.data_.groupby(dimension_bin, as_index=False)
    for variable in cst_variables:
        data[variable] = group_data[variable].median() + group_data[
            variable
        ].std().multiply(np.random.randn(len(group_data)), axis=0)
    data = data.fillna(definer.data_[cst_variables].median())
    mp_labels = definer.model_from_data(data.values)
    label_groups = mp_labels.reshape(-1, 100)
    value_groups = data[dimension].values.reshape(-1, 100)
    max_label = mp_labels.max()
    min_label = mp_labels.min()

    def f_prop(min, max):
        def func(arr):
            ans = np.zeros(max - min + 1)
            value_list, count_list = np.unique(arr, return_counts=True)
            total = np.sum(count_list)
            for value, count in zip(value_list, count_list):
                ans[value - min] = count / total
            return ans

        return func

    prop_along_axis = np.apply_along_axis(f_prop(min_label, max_label), 1, label_groups)
    threshold = np.where(np.diff(np.argmax(prop_along_axis, axis=1)) != 0)[0]
    index = np.sort(
        np.argsort(np.diff(threshold, prepend=0) / threshold)[-n_model + 1 :]
    )
    threshold = np.median(value_groups[threshold[index]], axis=1)
    if return_fake_data:
        return threshold, data
    return threshold


# np.random.seed(42)
# n_col = 4
# n_row = int(1e4)
# a = np.random.rand(n_row, n_col)
# a.T[0] = -np.log(a.T[0])
# a.T[1] *= 100
# a.T[2] = np.exp(a.T[2])
# columns = ["col0", "col1", "col2", "col3"]
# df = pd.DataFrame(a, columns=columns)
# target = np.random.rand(n_row)
# target = np.where(target > ((a.T[1] + a.T[2]) / np.max(a.T[1] + a.T[2])) ** 0.2, 1, 0)
# df["target"] = target

# from sklearn.model_selection import train_test_split

# train_df, test_df = train_test_split(df, test_size=0.25)

# mpd = ModelPointDefiner(train_df, "target")

# n_estimation = 100
# mpd.fit_cluster(n_estimation)

# mpd.fit_EM()

# mpd.fit_BSG(columns=["col1", "col2"], normalization=["n", "n"])
