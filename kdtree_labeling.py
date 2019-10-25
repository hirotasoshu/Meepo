import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class KDTreeLabeling:

    def __init__(self, r, bound, stream_count, params, len_):
        self.r = r
        self.bound = bound
        self.stream_count = stream_count
        self.params = params
        self.len_ = len_

    def _calc_labels(self, tree, space):
        labels = np.array([-1] * self.stream_count * self.len_, dtype=int)
        classes = np.hstack(np.array([i] * self.len_) for i in range(self.stream_count))
        for idx in range(self.len_ * self.stream_count):
            if all(np.isfinite(space[idx])):
                nears_ids = tree.query_radius(space[idx].reshape(1, -1), r=self.r, return_distance=False)[0]
                other_points = nears_ids[classes[nears_ids] != classes[idx]]
                if other_points.shape[0] > self.bound:
                    labels[idx] = -1
                else:
                    labels[idx] = classes[idx]
        labels = self._calc_labels_for_mix_points(labels, space)
        return labels

    def _calc_labels_for_mix_points(self, labels, space):
        mixed_points = space[labels == -1]
        #
        if (len(mixed_points)):
            mixed_points = DBSCAN(eps=3, min_samples=2).fit(mixed_points)
            mixed_points_labels = np.array([(label+1)*(-1) for label in mixed_points.labels_])
            j = 0
            for i in range(len(labels)):
                if labels[i] == -1:
                    labels[i] = mixed_points_labels[j]
                    j += 1
        return labels


    def fit(self, space):
        if space.size == 0:
            raise ValueError("Empty space")
        if not self.r > 0.0:
            raise ValueError("r must be positive.")
        tree = KDTree(space)
        labels = self._calc_labels(tree, space)
        return labels


if __name__ == "__main__":
    dots = [np.random.rand(2) for i in range(100)]
    dots = np.array(dots)
    labeling = KDTreeLabeling(r=2e-1, bound=6, stream_count=5, params=2, len_=20)
    labels = labeling.fit(dots)
    plt.scatter([dots[0] for i in range(len(dots))], [dots[1] for i in range(len(dots))], c=labels)
    plt.show()
    print(labels)