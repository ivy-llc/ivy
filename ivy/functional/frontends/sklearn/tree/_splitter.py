import ivy

FEATURE_THRESHOLD = 1e-7


class Splitter:
    def __init__(
        self,
        criterion,
        max_features,
        min_samples_leaf,
        min_weight_leaf,
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


class DensePartitioner:
    X = []
    samples = []
    feature_values = []
    start = 0
    end = 0
    n_missing = 0
    missing_values_in_feature_mask = []

    def __init__(
        self,
        X,
        samples,
        feature_values,
        missing_values_in_feature_mask,
    ):
        self.X = X
        self.samples = samples
        self.feature_values = feature_values
        self.missing_values_in_feature_mask = missing_values_in_feature_mask

    def init_node_split(self, start, end):
        self.start = start
        self.end = end
        self.n_missing = 0

    def sort_samples_and_feature_values(self, current_feature):
        feature_values = self.feature_values
        X = self.X
        samples = self.samples
        n_missing = 0
        missing_values_in_feature_mask = self.missing_values_in_feature_mask
        if (
            missing_values_in_feature_mask is not None
            and missing_values_in_feature_mask[current_feature]
        ):
            i, current_end = self.start, self.end - 1
            while i <= current_end:
                if ivy.isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue
                if ivy.isnan(X[samples[i], current_feature]):
                    samples[i], samples[current_end] = samples[current_end], samples[i]
                    n_missing += 1
                    current_end -= 1
                feature_values[i] = X[samples[i], current_feature]
                i += 1
        else:
            for i in range(self.start, self.end):
                feature_values[i] = X[int(samples[i]), int(current_feature)]
        (
            self.feature_values[self.start : self.end],
            self.samples[self.start : self.end],
        ) = sort(
            feature_values[self.start : self.end],
            samples[self.start : self.end],
            self.end - self.start - n_missing,
        )
        self.n_missing = n_missing

    def find_min_max(
        self,
        current_feature: int,
        min_feature_value_out: float,
        max_feature_value_out: float,
    ):
        current_feature = 0
        X = self.X
        samples = self.samples
        min_feature_value = X[samples[self.start], current_feature]
        max_feature_value = min_feature_value
        feature_values = self.feature_values
        feature_values[self.start] = min_feature_value
        for p in range(self.start + 1, self.end):
            current_feature_value = X[samples[p], current_feature]
            feature_values[p] = current_feature_value

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value
        return min_feature_value, max_feature_value

    def next_p(self, p_prev: int, p: int):
        feature_values = self.feature_values
        end_non_missing = self.end - self.n_missing

        while (
            p + 1 < end_non_missing
            and feature_values[p + 1] <= feature_values[p] + FEATURE_THRESHOLD
        ):
            p += 1
        p_prev = p
        p += 1
        return p_prev, p

    def partition_samples(self, current_thershold: float):
        p = self.start
        partition_end = self.end
        samples = self.samples
        feature_values = self.feature_values
        while p < partition_end:
            if feature_values[p] <= current_thershold:
                p += 1
            else:
                partition_end -= 1

                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end],
                    feature_values[p],
                )
                samples[p], samples[partition_end] = (
                    samples[partition_end],
                    samples[p],
                )
        return partition_end

    def partition_samples_final(
        self,
        best_pos,
        best_threshold,
        best_feature,
        best_n_missing,
    ):
        start = self.start
        p = start
        end = self.end - 1
        partition_end = end - best_n_missing
        samples = self.samples
        X = self.X

        if best_n_missing != 0:
            while p < partition_end:
                if ivy.isnan(X[samples[end], best_feature]):
                    end -= 1
                    continue
                current_value = X[samples[p], best_feature]
                if ivy.isnan(current_value):
                    samples[p], samples[end] = samples[end], samples[p]
                    end -= 1
                    current_value = X[samples[p], best_feature]
                if current_value <= best_threshold:
                    p += 1
                else:
                    samples[p], samples[partition_end] = (
                        samples[partition_end],
                        samples[p],
                    )
                    partition_end -= 1
        else:
            while p < partition_end:
                if X[samples[p], best_feature] <= best_threshold:
                    p += 1
                else:
                    samples[p], samples[partition_end] = (
                        samples[partition_end],
                        samples[p],
                    )
                    partition_end -= 1
        self.samples = samples


class BestSplitter(Splitter):
    def init(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        *args,
    ):
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask, *args)
        self.partitioner = None

    def node_split(self, impurity, split, n_constant_features):
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            impurity,
            split,
            n_constant_features,
        )


def node_split_best(
    splitter: Splitter, partitioner, criterion, impurity, split, n_constant_features
):
    pass


def sort(feature_values, samples, n):
    return 0, 0
