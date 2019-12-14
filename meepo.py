import numpy as np
from sklearn.neighbors import KDTree


class Meepo:

    def __init__(self, eps=1, min_samples=2):
        self._check_eps(eps)
        self._check_min_samples(min_samples)
        self.eps = eps
        self.min_samples = min_samples

    @staticmethod
    def _check_eps(eps):
        if eps < 0.0:
            raise ValueError("Epsilon must be positive.")

    @staticmethod
    def _check_min_samples(min_samples):
        if min_samples < 0:
            raise ValueError("Min samples must be positive")

    @staticmethod
    def _check_data_and_labels(space, classes):
        if not np.all(np.isfinite(space)) and space.size:
            raise ValueError("The data is empty or contains NaN's")
        elif not np.all(np.isfinite(classes)) and classes.size:
            raise ValueError("The labels is empty or contains NaN's")
        elif len(space) != len(classes):
            raise ValueError("Data size must be equal to classes size")

    def _get_labels(self, tree, data, labels):
        shape = (len(labels), )
        predicted_labels = np.full(shape, -1, dtype=int)
        return self._get_mixed_points(labels, predicted_labels, data, tree)

    def _get_mixed_points(self, labels, predicted_labels, data, tree):
        for idx, class_ in enumerate(labels):
            nears_ids = tree.query_radius(data[idx].reshape(1, -1), r=self.eps, return_distance=False)[0]
            other_points = nears_ids[labels[nears_ids] != class_]
            predicted_labels[idx] = -1 if other_points.shape[0] > self.min_samples else class_
        return predicted_labels

    def fit_predict(self, data, labels, metric='euclidean'):
        self._check_data_and_labels(data, labels)
        tree = KDTree(data, metric=metric)
        return self._get_labels(tree, data, labels)
