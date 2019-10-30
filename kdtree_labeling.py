import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN


class KDTreeLabeling:

    def __init__(self, r, bound):
        self.r = r
        self.bound = bound
        self._check_r()
        self._check_bound()

    def _check_r(self):
        if not self.r >= 0.0:
            raise ValueError("R must be positive.")

    def _check_bound(self):
        if not self.bound >= 0:
            raise ValueError("Bound must be positive")

    @staticmethod
    def _check_space_and_classes(space, classes):
        if not np.all(np.isfinite(space)) and space.size:
            raise ValueError("The space is empty or contains NaN's")
        if not np.all(np.isfinite(classes)) and classes.size:
            raise ValueError("The classes is empty or contains NaN's")
        if len(space) != len(classes):
            raise ValueError("Space size must be equal to classes size")

    def _get_labels(self, tree, space, classes):
        labels = np.array([-1] * len(classes), dtype=int)
        labels = self._get_mixed_points(classes, labels, space, tree)
        labels = self._get_labels_for_mix_points(labels, space)
        return labels

    def _get_mixed_points(self, classes, labels, space, tree):
        for idx in range(len(classes)):
            nears_ids = tree.query_radius(space[idx].reshape(1, -1), r=self.r, return_distance=False)[0]
            other_points = nears_ids[classes[nears_ids] != classes[idx]]
            if other_points.shape[0] > self.bound:
                labels[idx] = -1
            else:
                labels[idx] = classes[idx]
        return labels

    @staticmethod
    def _get_labels_for_mix_points(labels, space):
        mixed_points = space[labels == -1]
        if mixed_points.size:
            mixed_points_labels = DBSCAN(eps=3, min_samples=2).fit_predict(mixed_points)
            mixed_points_labels = np.array([(label + 1)*(-1) for label in mixed_points_labels])
            j = 0
            for i in range(len(labels)):
                if labels[i] == -1:
                    labels[i] = mixed_points_labels[j]
                    j += 1
        return labels

    def fit_predict(self, space, classes):
        self._check_space_and_classes(space, classes)
        tree = KDTree(space)
        labels = self._get_labels(tree, space, classes)
        return labels

