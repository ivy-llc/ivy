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


class SplitRecord:
    def __init__(
        self,
        feature=0,
        pos=0,
        threshold=0.0,
        improvement=-ivy.inf,
        impurity_left=0.0,
        impurity_right=0.0,
        missing_go_to_left=False,
        n_missing=0,
    ):
        self.feature = feature
        self.pos = pos
        self.threshold = threshold
        self.improvement = improvement
        self.impurity_left = impurity_left
        self.impurity_right = impurity_right
        self.missing_go_to_left = missing_go_to_left
        self.n_missing = n_missing


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
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    def node_split(self, impurity, split, n_constant_features):
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            impurity,
            split,
            n_constant_features,
        )


# --- Helpers --- #
# --------------- #


def _init_split(split_record, start_pos):
    split_record.impurity_left = ivy.inf
    split_record.impurity_right = ivy.inf
    split_record.pos = start_pos
    split_record.feature = 0
    split_record.threshold = 0.0
    split_record.improvement = -ivy.inf
    split_record.missing_go_to_left = False
    split_record.n_missing = 0
    return split_record


# --- Main --- #
# ------------ #


def node_split_best(
    splitter, partitioner, criterion, impurity, split, n_constant_features
):
    start = splitter.start
    end = splitter.end
    features = splitter.features
    constant_features = splitter.constant_features
    n_features = splitter.n_features

    feature_values = splitter.feature_values
    max_features = splitter.max_features
    min_samples_leaf = splitter.min_samples_leaf
    min_weight_leaf = splitter.min_weight_leaf

    best_split = SplitRecord()
    current_split = SplitRecord()
    best_proxy_improvement = -ivy.inf

    f_i = n_features
    p_prev = 0

    n_visited_features = 0
    # Number of features discovered to be constant during the split search
    n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    n_drawn_constants = 0
    n_known_constants = n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    n_total_constants = n_known_constants
    best_split = _init_split(best_split, end)
    partitioner.init_node_split(start, end)
    while f_i > n_total_constants and (
        n_visited_features < max_features
        or n_visited_features <= n_found_constants + n_drawn_constants
    ):
        n_visited_features += 1
        f_j = ivy.randint(n_drawn_constants, f_i - n_found_constants)

        if f_j < n_known_constants:
            features[n_drawn_constants], features[f_j] = (
                features[f_j],
                features[n_drawn_constants],
            )

            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[
        current_split.feature = features[f_j]
        partitioner.sort_samples_and_feature_values(current_split.feature)
        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (
            end_non_missing == start
            or feature_values[end_non_missing - 1]
            <= feature_values[start] + FEATURE_THRESHOLD
        ):
            features[f_j], features[n_total_constants] = (
                features[n_total_constants],
                features[f_j],
            )

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0
        criterion.init_missing(n_missing)
        n_searches = 2 if has_missing else 1
        for i in range(n_searches):
            missing_go_to_left = i == 1
            criterion.missing_go_to_left = missing_go_to_left
            criterion.reset()
            p = start

            while p < end_non_missing:
                p_prev, p = partitioner.next_p(p_prev, p)

                if p >= end_non_missing:
                    continue

                if missing_go_to_left:
                    n_left = p - start + n_missing
                    n_right = end_non_missing - p
                else:
                    n_left = p - start
                    n_right = end_non_missing - p + n_missing

                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                current_split.pos = p
                criterion.update(current_split.pos)

                if (
                    criterion.weighted_n_left < min_weight_leaf
                    or criterion.weighted_n_right < min_weight_leaf
                ):
                    continue

                current_proxy_improvement = criterion.proxy_impurity_improvement()

                if current_proxy_improvement > best_proxy_improvement:
                    best_proxy_improvement = current_proxy_improvement
                    current_split.threshold = (
                        feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                    )

                    if current_split.threshold in (
                        feature_values[p],
                        ivy.inf,
                        -ivy.inf,
                    ):
                        current_split.threshold = feature_values[p_prev]

                    current_split.n_missing = n_missing
                    if n_missing == 0:
                        current_split.missing_go_to_left = n_left > n_right
                    else:
                        current_split.missing_go_to_left = missing_go_to_left

                    best_split = SplitRecord(**current_split.__dict__)

        if has_missing:
            n_left, n_right = end - start - n_missing, n_missing
            p = end - n_missing
            missing_go_to_left = 0

            if not ((n_left < min_samples_leaf) or (n_right < min_samples_leaf)):
                criterion.missing_go_to_left = missing_go_to_left
                criterion.update(p)

                if not (
                    criterion.weighted_n_left < min_weight_leaf
                    or criterion.weighted_n_right < min_weight_leaf
                ):
                    current_proxy_improvement = criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current_split.threshold = ivy.inf
                        current_split.missing_go_to_left = missing_go_to_left
                        current_split.n_missing = n_missing
                        current_split.pos = p
                        best_split = current_split

    # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
    if best_split.pos < end:
        partitioner.partition_samples_final(
            best_split.pos,
            best_split.threshold,
            best_split.feature,
            best_split.n_missing,
        )

        if best_split.n_missing != 0:
            criterion.init_missing(best_split.n_missing)

        criterion.missing_go_to_left = best_split.missing_go_to_left
        criterion.reset()
        criterion.update(best_split.pos)

        (
            best_split.impurity_left,
            best_split.impurity_right,
        ) = criterion.children_impurity(
            best_split.impurity_left, best_split.impurity_right
        )

        best_split.improvement = criterion.impurity_improvement(
            impurity, best_split.impurity_left, best_split.impurity_right
        )

        # best_split, samples = shift_missing_values_to_left_if_required(
        # best_split, samples, end)
        # todo : implement shift_missing_values_to_left_if_required
    features[0:n_known_constants] = constant_features[0:n_known_constants]
    constant_features[n_known_constants:n_found_constants] = features[
        n_known_constants:n_found_constants
    ]

    split = best_split
    n_constant_features = n_total_constants
    return 0, n_constant_features, split


def sort(feature_values, samples, n):
    if n == 0:
        return
    idx = ivy.argsort(feature_values)
    return feature_values[idx], samples[idx]
