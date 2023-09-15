# splitter.init
# splitter.n_samples
# splitter.node_reset
# splitter.node_impurity
# splitter.node_split
# splitter.node_value

from ._criterion import Criterion
from ctypes import Union
import ivy
import random

FEATURE_THRESHOLD = 1e-7


class SplitRecord:
    def __init__():
        raise NotImplementedError(
            "No idea what SplitRecord is, just created to remove error!"
        )


class Splitter:
    """
    Abstract splitter class.

    Splitters are called by tree builders to find the best splits on
    both sparse and dense data, one split at a time.
    """

    def __init__(
        self, criterion, max_features, min_samples_leaf, min_weight_leaf, random_state
    ):
        self.criterion = criterion
        self.random_state = random_state

        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf

        self.rand_r_state = random_state.randint(0, 0x7FFFFFFF)
        self.samples = None
        self.weighted_n_samples = 0.0

        self.features = None
        self.feature_values = None
        self.constant_features = None

        self.y = None
        self.sample_weight = None

    def init(self, X, y, sample_weight, missing_values_in_feature_mask=None):
        n_samples = X.shape[0]

        # Create a list to store indices of positively weighted samples
        samples = []

        weighted_n_samples = 0.0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples.append(i)

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Set the attributes
        self.n_samples = len(samples)
        self.weighted_n_samples = weighted_n_samples

        n_features = X.shape[1]
        self.features = ivy.arange(n_features)
        self.n_features = n_features

        self.samples = ivy.array(samples, dtype="int32")
        self.feature_values = ivy.empty(n_samples, dtype="float32")
        self.constant_features = ivy.empty(n_features, dtype="int32")

        self.y = y
        self.sample_weight = sample_weight

        if missing_values_in_feature_mask is not None:
            self.criterion.init_sum_missing()

        return 0

    def __reduce__(self):
        return (
            type(self),
            (
                self.criterion,
                self.max_features,
                self.min_samples_leaf,
                self.min_weight_leaf,
                self.random_state,
            ),
            self.__getstate__(),
        )

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

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    def node_split(self, impurity, split, n_constant_features):
        # This is a placeholder method; actual implementation required.
        # You should add your computation logic here to find the best split.
        pass

    def node_value(self, dest):
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)

    def node_impurity(self):
        """Return the impurity of the current node."""
        return self.criterion.node_impurity()


class DensePartitioner:
    """
    Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy
    (best vs. random).
    """

    def __init__(self, X, samples, feature_values, missing_values_in_feature_mask):
        self.X = X
        self.samples = samples
        self.feature_values = feature_values
        self.missing_values_in_feature_mask = missing_values_in_feature_mask

    def init_node_split(self, start, end):
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.n_missing = 0

    def sort_samples_and_feature_values(self, current_feature):
        i = 0
        current_end = 0
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
                feature_values[i] = X[samples[i], current_feature]

        # Sorting algorithm not shown here; implement the sorting logic separately.

        self.n_missing = n_missing


class SparsePartitioner:
    def __init__(
        self,
        X,
        samples,
        n_samples,
        feature_values,
        missing_values_in_feature_mask,
    ):
        if not self.isspmatrix_csc(X):
            raise ValueError("X should be in csc format")

        self.samples = samples
        self.feature_values = feature_values

        # Initialize X
        n_total_samples = X.shape[0]

        self.X_data = X.data
        self.X_indices = X.indices
        self.X_indptr = X.indptr
        self.n_total_samples = n_total_samples

        # Initialize auxiliary arrays used for partitioning
        self.index_to_samples = ivy.full(n_total_samples, fill_value=-1, dtype=ivy.intp)
        self.sorted_samples = ivy.empty(n_samples, dtype=ivy.intp)
        self.n_missing = 0  # Placeholder for missing values (not supported yet)
        self.missing_values_in_feature_mask = missing_values_in_feature_mask

        self.start_positive = 0
        self.end_negative = 0
        self.is_samples_sorted = False

    def init_node_split(self, start, end):
        self.start = start
        self.end = end
        self.is_samples_sorted = False

    def sort_samples_and_feature_values(self, current_feature):
        # Simultaneously sort based on feature_values
        self.extract_nnz(current_feature)
        # Rest of the code for sorting...

    def find_min_max(
        self, current_feature, min_feature_value_out, max_feature_value_out
    ):
        # Find the minimum and maximum value for current_feature
        self.extract_nnz(current_feature)
        # Rest of the code for finding min and max...

    def next_p(self, p_prev, p):
        # Compute the next p_prev and p for iterating over feature values
        # Rest of the code...
        pass

    def partition_samples(self, current_threshold):
        # Partition samples for feature_values at the current_threshold
        return self._partition(current_threshold, self.start_positive)

    def partition_samples_final(
        self, best_pos, best_threshold, best_feature, n_missing
    ):
        # Partition samples for X at the best_threshold and best_feature
        self.extract_nnz(best_feature)
        self._partition(best_threshold, best_pos)

    def _partition(self, threshold, zero_pos):
        # Partition samples based on threshold
        # Rest of the code...
        pass

    def extract_nnz(self, feature):
        # Extract and partition values for a given feature
        # Rest of the code...
        pass

    def isspmatrix_csc(self, X):
        # Check if X is in CSC format
        # You may need to implement this function according to your specific requirements
        # For the example, we assume it's already in CSC format
        return True


class BestSplitter(Splitter):
    """Splitter for finding the best split on dense data."""

    def __init__(self, X, y, sample_weight, missing_values_in_feature_mask):
        super().__init__(X, y, sample_weight, missing_values_in_feature_mask)
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


class BestSparseSplitter(Splitter):
    """Splitter for finding the best split, using sparse data."""

    def __init__(self, X, y, sample_weight, missing_values_in_feature_mask):
        super().__init__(X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X,
            self.samples,
            self.n_samples,
            self.feature_values,
            missing_values_in_feature_mask,
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


class RandomSplitter(Splitter):
    """Splitter for finding the best random split on dense data."""

    def __init__(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
    ):
        Splitter.__init__(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    def node_split(self, impurity, split, n_constant_features):
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            impurity,
            split,
            n_constant_features,
        )


class RandomSparseSplitter(Splitter):
    """Splitter for finding the best random split, using sparse data."""

    def __init__(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
    ):
        Splitter.__init__(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X,
            self.samples,
            self.n_samples,
            self.feature_values,
            missing_values_in_feature_mask,
        )

    def node_split(self, impurity, split, n_constant_features):
        return node_split_random(
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
    split_record.impurity_left = INFINITY
    split_record.impurity_right = INFINITY
    split_record.pos = start_pos
    split_record.feature = 0
    split_record.threshold = 0.0
    split_record.improvement = -INFINITY
    split_record.missing_go_to_left = False
    split_record.n_missing = 0


# --- Main --- #
# ------------ #


def heapsort(feature_values, samples, n):
    # Heapify
    start = (n - 2) // 2
    end = n
    while True:
        sift_down(feature_values, samples, start, end)
        if start == 0:
            break
        start -= 1

    # Sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(feature_values, samples, 0, end)
        sift_down(feature_values, samples, 0, end)
        end -= 1


def introsort(feature_values, samples, n, maxd):
    while n > 1:
        if maxd <= 0:  # max depth limit exceeded ("gone quadratic")
            # Implement or import heapsort function
            heapsort(feature_values, samples, n)
            return
        maxd -= 1

        pivot = median3(feature_values, n)

        # Three-way partition.
        i = j = 0
        r = n
        while i < r:
            if feature_values[i] < pivot:
                swap(feature_values, samples, i, j)
                i += 1
                j += 1
            elif feature_values[i] > pivot:
                r -= 1
                swap(feature_values, samples, i, r)
            else:
                i += 1

        introsort(feature_values[:j], samples[:j], j, maxd)
        feature_values = feature_values[r:]
        samples = samples[r:]
        n -= r


def median3(feature_values, n):
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    a, b, c = feature_values[0], feature_values[n // 2], feature_values[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


def node_split_best(
    splitter: Splitter,
    partitioner: Union[DensePartitioner, SparsePartitioner],
    criterion: Criterion,
    impurity: float,
    split: SplitRecord,
    n_constant_features: int,
):
    """
    Find the best split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise
    MemoryError) or 0 otherwise.
    """
    # Find the best split
    start = splitter.start
    end = splitter.end
    end_non_missing = 0
    n_missing = 0
    has_missing = 0
    n_searches = 0
    n_left = 0
    n_right = 0
    missing_go_to_left = 0

    samples = splitter.samples
    features = splitter.features
    constant_features = splitter.constant_features
    n_features = splitter.n_features

    feature_values = splitter.feature_values
    max_features = splitter.max_features
    min_samples_leaf = splitter.min_samples_leaf
    min_weight_leaf = splitter.min_weight_leaf
    random_state = random.Random(splitter.rand_r_state)

    best_split = SplitRecord()
    current_split = SplitRecord()
    current_proxy_improvement = -INFINITY
    best_proxy_improvement = -INFINITY

    f_i = n_features
    f_j = 0
    p = 0
    p_prev = 0

    n_visited_features = 0
    # Number of features discovered to be constant during the split search
    n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    n_drawn_constants = 0
    n_known_constants = n_constant_features[0]
    # n_total_constants = n_known_constants + n_found_constants
    n_total_constants = n_known_constants

    _init_split(best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while f_i > n_total_constants and (
        n_visited_features < max_features
        or n_visited_features <= n_found_constants + n_drawn_constants
    ):
        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = ivy.random.randint(
            n_drawn_constants, f_i - n_found_constants, random_state
        )

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
            # All values for this feature are missing, or
            end_non_missing == start
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            or feature_values[end_non_missing - 1]
            <= feature_values[start] + FEATURE_THRESHOLD
        ):
            # We consider this feature constant in this case.
            # Since finding a split among constant feature is not valuable,
            # we do not consider this feature for splitting.
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

        if has_missing:
            criterion.init_missing(n_missing)

        # Evaluate all splits

        # If there are missing values, then we search twice for the most optimal split.
        # The first search will have all the missing values going to the right node.
        # The second search will have all the missing values going to the left node.
        # If there are no missing values, then we search only once for the most
        # optimal split.
        n_searches = 2 if has_missing else 1

        for i in range(n_searches):
            missing_go_to_left = i == 1
            criterion.missing_go_to_left = missing_go_to_left
            criterion.reset()
            p = start

            while p < end_non_missing:
                partitioner.next_p(p_prev, p)

                if p >= end_non_missing:
                    continue

                if missing_go_to_left:
                    n_left = p - start + n_missing
                    n_right = end_non_missing - p
                else:
                    n_left = p - start
                    n_right = end_non_missing - p + n_missing

                # Reject if min_samples_leaf is not guaranteed
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                current_split.pos = p
                criterion.update(current_split.pos)

                # Reject if min_weight_leaf is not satisfied
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

                    if (
                        current_split.threshold == feature_values[p]
                        or current_split.threshold == INFINITY
                        or current_split.threshold == -INFINITY
                    ):
                        current_split.threshold = feature_values[p_prev]

                    current_split.n_missing = n_missing
                    if n_missing == 0:
                        current_split.missing_go_to_left = n_left > n_right
                    else:
                        current_split.missing_go_to_left = missing_go_to_left

                    best_split = current_split

        # Evaluate when there are missing values and all missing values goes
        # to the right node and non-missing values goes to the left node.
        if has_missing:
            n_left, n_right = end - start - n_missing, n_missing
            p = end - n_missing
            missing_go_to_left = 0

            if not (
                (criterion.weighted_n_left < min_samples_leaf)
                or (criterion.weighted_n_right < min_samples_leaf)
            ):
                criterion.missing_go_to_left = missing_go_to_left
                criterion.update(p)

                if not (
                    criterion.weighted_n_left < min_weight_leaf
                    or criterion.weighted_n_right < min_weight_leaf
                ):
                    current_proxy_improvement = criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current_split.threshold = INFINITY
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

        criterion.children_impurity(best_split.impurity_left, best_split.impurity_right)

        best_split.improvement = criterion.impurity_improvement(
            impurity, best_split.impurity_left, best_split.impurity_right
        )

        shift_missing_values_to_left_if_required(best_split, samples, end)

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    features[:n_known_constants] = constant_features[:n_known_constants]
    constant_features += features[
        n_known_constants : n_known_constants + n_found_constants
    ]

    # Return Values
    split[0] = best_split
    n_constant_features[0] = n_total_constants
    return 0


def node_split_random(
    splitter: Splitter,
    partitioner: Union[DensePartitioner, SparsePartitioner],
    criterion: Criterion,
    impurity: float,
    split: SplitRecord,
    n_constant_features: int,
):
    """
    Find the best random split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise
    MemoryError) or 0 otherwise.
    """
    # Draw random splits and pick the best
    start = splitter.start
    end = splitter.end

    features = splitter.features
    constant_features = splitter.constant_features
    n_features = splitter.n_features

    max_features = splitter.max_features
    min_samples_leaf = splitter.min_samples_leaf
    min_weight_leaf = splitter.min_weight_leaf
    random_state = random.Random(splitter.rand_r_state)

    best_split = SplitRecord()
    current_split = SplitRecord()
    current_proxy_improvement = -INFINITY
    best_proxy_improvement = -INFINITY

    f_i = n_features
    f_j = 0
    # Number of features discovered to be constant during the split search
    n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    n_drawn_constants = 0
    n_known_constants = n_constant_features[0]
    # n_total_constants = n_known_constants + n_found_constants
    n_total_constants = n_known_constants
    n_visited_features = 0
    min_feature_value = 0
    max_feature_value = 0

    _init_split(best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while f_i > n_total_constants and (
        n_visited_features < max_features
        or n_visited_features <= n_found_constants + n_drawn_constants
    ):
        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = ivy.random.randint(
            n_drawn_constants, f_i - n_found_constants, random_state
        )

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
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

        # Find min, max
        partitioner.find_min_max(
            current_split.feature, min_feature_value, max_feature_value
        )

        if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
            features[f_j], features[n_total_constants] = (
                features[n_total_constants],
                current_split.feature,
            )

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]

        # Draw a random thershold
        current_split.threshold = ivy.random.random_uniform(
            min_feature_value,
            max_feature_value,
            random_state,
        )

        if current_split.threshold == max_feature_value:
            current_split.threshold = min_feature_value

        # Partition
        current_split.pos = partitioner.partition_samples(current_split.threshold)

        # Reject if min_samples_leaf is not guaranteed
        if ((current_split.pos - start) < min_samples_leaf) or (
            (end - current_split.pos) < min_samples_leaf
        ):
            continue

        # Evaluate split
        # At this point, the criterion has a view into the samples that was partitioned
        # by the partitioner. The criterion will use the partition to evaluating the split.
        criterion.reset()
        criterion.update(current_split.pos)

        # Reject if min_weight_leaf is not satisfied
        if (criterion.weighted_n_left < min_weight_leaf) or (
            criterion.weighted_n_right < min_weight_leaf
        ):
            continue

        current_proxy_improvement = criterion.proxy_impurity_improvement()

        if current_proxy_improvement > best_proxy_improvement:
            best_proxy_improvement = current_proxy_improvement
            best_split = current_split  # copy

    # Reorganize into samples[start:best.pos] + samples[best.pos:end]
    if best_split.pos < end:
        if current_split.feature != best_split.feature:
            # TODO: Pass in best.n_missing when random splitter supports missing values.
            partitioner.partition_samples_final(
                best_split.pos, best_split.threshold, best_split.feature, 0
            )

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(best_split.impurity_left, best_split.impurity_right)
        best_split.improvement = criterion.impurity_improvement(
            impurity, best_split.impurity_left, best_split.impurity_right
        )

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    features[:n_known_constants] = constant_features[:n_known_constants]

    # Copy newly found constant features
    constant_features += features[
        n_known_constants : n_known_constants + n_found_constants
    ]

    # Return Values
    split[0] = best_split
    n_constant_features[0] = n_total_constants
    return 0


def shift_missing_values_to_left_if_required(best, samples, end):
    # The partitioner partitions the data such that the missing values are in
    # samples[-n_missing:] for the criterion to consume. If the missing values
    # are going to the right node, then the missing values are already in the
    # correct position. If the missing values go left, then we move the missing
    # values to samples[best.pos:best.pos+n_missing] and update `best.pos`.
    if best.n_missing > 0 and best.missing_go_to_left:
        for p in range(best.n_missing):
            i = best.pos + p
            current_end = end - 1 - p
            samples[i], samples[current_end] = samples[current_end], samples[i]
        best.pos += best.n_missing


def sift_down(feature_values, samples, start, end):
    # Restore heap order in feature_values[start:end] by moving the max element to start.
    root = start
    while True:
        child = root * 2 + 1

        # Find the max of root, left child, right child
        maxind = root
        if (
            child < end
            and feature_values[samples[maxind]] < feature_values[samples[child]]
        ):
            maxind = child
        if (
            child + 1 < end
            and feature_values[samples[maxind]] < feature_values[samples[child + 1]]
        ):
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(feature_values, samples, root, maxind)
            root = maxind


def sort(feature_values, samples, n):
    if n == 0:
        return
    maxd = 2 * int(ivy.log(n))
    introsort(feature_values, samples, n, maxd)


def swap(feature_values, samples, i, j):
    feature_values[samples[i]], feature_values[samples[j]] = (
        feature_values[samples[j]],
        feature_values[samples[i]],
    )
    samples[i], samples[j] = samples[j], samples[i]


INFINITY = ivy.inf
