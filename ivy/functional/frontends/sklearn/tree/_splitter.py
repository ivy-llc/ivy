import ivy


class Splitter:
    def __init__(
        self,
        criterion,
        max_features: int,
        min_samples_leaf: int,
        min_weight_leaf: float,
        random_state,
        *args,
    ):
        self.criterion = criterion
        self.n_samples = 0
        self.n_features = 0
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

    def init(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        *args,
    ):

        n_samples = X.shape[0]
        self.samples = ivy.empty(n_samples, dtype=ivy.int32)
        samples = self.samples
        j = 0
        weighted_n_samples = 0.0

        for i in range(n_samples):
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1
            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples
        n_features = X.shape[1]
        self.features = ivy.arange(n_features, dtype=ivy.int32)
        self.n_features = n_features
        self.feature_values = ivy.empty(n_samples, dtype=ivy.float32)
        self.constant_features = ivy.empty(n_features, dtype=ivy.int32)
        self.y = y
        self.sample_weight = sample_weight
        if missing_values_in_feature_mask is not None:
            self.criterion.init_sum_missing()
        return 0

    def node_reset(self, start, end, weighted_n_node_samples):
        self.start = start
        self.end = end
        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end,
        )
        weighted_n_node_samples = self.criterion.weighted_n_node_samples
        return 0, weighted_n_node_samples

    def node_split(self, impurity, split, n_constant_features):
        pass

    def node_value(self, dest, node_id):
        return self.criterion.node_value(dest, node_id)

    def node_impurity(self):
        return self.criterion.node_impurity()
