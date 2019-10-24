import numpy as np
from sklearn.neighbors import KDTree


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
        return labels

    def fit(self, space):
        if space.size == 0:
            raise ValueError("Empty space")
        if not self.r > 0.0:
            raise ValueError("r must be positive.")
        tree = KDTree(space)
        labels = self._calc_labels(tree, space)
        return labels



labeling = KDTreeLabeling(r=1, bound=0, stream_count=1, params=3, len_=2)
space = np.array([[0, 0, 0], [1, 1, 1]])
print(labeling.fit(space))

