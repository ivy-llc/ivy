from ._criterion import Criterion
import ivy

from ._utils import rand_int
from ._utils import RAND_R_MAX
import random

INFINITY = ivy.inf

# Mitigate precision differences between 32 bit and 64 bit
FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparsePartitioner
EXTRACT_NNZ_SWITCH = 0.1


class SplitRecord:
    def __init__(
        self,
        feature=0,
        pos=0,
        threshold=0.0,
        improvement=-INFINITY,
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


def _init_split(split_record, start_pos):
    split_record.impurity_left = INFINITY
    split_record.impurity_right = INFINITY
    split_record.pos = start_pos
    split_record.feature = 0
    split_record.threshold = 0.0
    split_record.improvement = -INFINITY
    split_record.missing_go_to_left = False
    split_record.n_missing = 0


class Splitter:
    """
    Abstract splitter class.

    Splitters are called by tree builders to find the best splits on
    both sparse and dense data, one split at a time.
    """

    def __init__(
        self,
        criterion: Criterion,
        max_features: int,
        min_samples_leaf: int,
        min_weight_leaf: float,
        random_state,
        *args
    ):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features :
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf :
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """
        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

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

    def init(
        self, X, y: list, sample_weight: list, missing_values_in_feature_mask: list, *args
    ):
        """
        Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples represented
            as a Cython memoryview.

        sample_weight : ndarray, dtype=DOUBLE_t
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight. This is represented
            as a Cython memoryview.

        has_missing : bool
            At least one missing values is in X.
        """
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)

        n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        self.samples = ivy.empty(n_samples, dtype=ivy.int32)
        samples = self.samples

        i = 0
        j = 0
        weighted_n_samples = 0.0
        
        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
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

    def node_reset(self, start: int, end: int, weighted_n_node_samples: list):
        """
        Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start :
            The index of the first sample to consider
        end :
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=double pointer
            The total weight of those samples
        """

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

    def node_split(
        self, impurity: float, split: SplitRecord, n_constant_features: list
    ):
        """
        Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will
        be done here.

        It should return -1 upon errors.
        """
        pass

    def node_value(self, dest):
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)

    def node_impurity(self):
        """Return the impurity of the current node."""
        return self.criterion.node_impurity()


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


def node_split_best(
    splitter: Splitter,
    partitioner,
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
    random_state = splitter.rand_r_state

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
    n_known_constants = n_constant_features
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
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)

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
    features[0:n_known_constants * 4] = constant_features[0:n_known_constants * 4]
    # Copy newly found constant features
    constant_features[n_known_constants:n_found_constants * 4] = features[n_known_constants:n_found_constants * 4]

    # Return Values
    split[0] = best_split
    n_constant_features[0] = n_total_constants
    return 0, n_constant_features


def sort(feature_values, samples, n):
    print("---sort---")
    print(f"feature_values: {feature_values}")
    print(f"samples: {samples}")
    print(f"n: {n}")
    print("---sort---")
    if n == 0:
        return
    maxd = 2 * int(ivy.log(n))
    introsort(feature_values, samples, n, maxd)


def swap(feature_values, samples, i, j):
    print("---swap---")
    print(f"feature_values: {feature_values}")
    print(f"samples: {samples}")
    print(f"i: {i}")
    print(f"j: {j}")
    print("---swap---")
    feature_values[samples[i]], feature_values[samples[j]] = (
        feature_values[samples[j]],
        feature_values[samples[i]],
    )
    samples[i], samples[j] = samples[j], samples[i]


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


def introsort(feature_values, samples, n, maxd):
    print("---introsort---")
    print(f"feature_values: {feature_values}")
    print(f"samples: {samples}")
    print(f"n: {n}")
    print(f"maxd: {maxd}")
    print("---introsort---")
    while n > 1:
        if maxd <= 0:  # max depth limit exceeded ("gone quadratic")
            # Implement or import heapsort function
            heapsort(feature_values, samples, n)
            return
        maxd -= 1

        pivot = median3(feature_values, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if feature_values[i] < pivot:
                swap(feature_values, samples, i, l)
                i += 1
                l += 1
            elif feature_values[i] > pivot:
                r -= 1
                swap(feature_values, samples, i, r)
            else:
                i += 1

        introsort(feature_values, samples, l, maxd)
        feature_values = feature_values[r:]
        samples = samples[r:]
        n -= r


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



def heapsort(feature_values, samples, n):
    print("---heapsort---")
    print(f"feature_values: {feature_values}")
    print(f"samples {samples}")
    print(f"n: {n}")
    print("---heapsort---")
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


def node_split_random(
    splitter: Splitter,
    partitioner,
    criterion: Criterion,
    impurity: float,
    split: SplitRecord,
    n_constant_features: list,
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
    while (
        f_i > n_total_constants
        and  # Stop early if remaining features
        # are constant
        (
            n_visited_features < max_features
            or
            # At least one drawn features must be non constant
            n_visited_features <= n_found_constants + n_drawn_constants
        )
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
        f_j = random.randint(n_drawn_constants, f_i - n_found_constants)

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

        # Draw a random threshold
        current_split.threshold = random.uniform(
            min_feature_value,
            max_feature_value,
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
    features[0 : n_known_constants * 4] = constant_features[0 : n_known_constants * 4]

    # Copy newly found constant features
    constant_features[n_known_constants : n_found_constants * 4] = features[
        n_known_constants : n_found_constants * 4
    ]

    # Return values
    split[0] = best_split
    n_constant_features[0] = n_total_constants
    return 0, n_constant_features


class DensePartitioner:
    """
    Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy
    (best vs. random).
    """

    X = []
    samples = []
    feature_values = []
    start = 0
    end = 0
    n_missing = 0
    missing_values_in_feature_mask = []

    def __init__(
        self,
        X: list,
        samples: list,
        feature_values: list,
        missing_values_in_feature_mask: list,
    ):
        self.X = X
        self.samples = samples
        self.feature_values = feature_values
        self.missing_values_in_feature_mask = missing_values_in_feature_mask

    def init_node_split(self, start: int, end: int):
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.n_missing = 0

    def sort_samples_and_feature_values(self, current_feature: int):
        print("---sort_samples_and_feature_values---")
        print(f"current_feature: {current_feature}")
        print("---sort_samples_and_feature_values---")
        """
        Simultaneously sort based on the feature_values.

        Missing values are stored at the end of feature_values. The
        number of missing values observed in feature_values is stored in
        self.n_missing.
        """
        i = 0
        current_end = 0
        feature_values = self.feature_values
        X = self.X
        samples = self.samples
        n_missing = 0
        missing_values_in_feature_mask = self.missing_values_in_feature_mask

        # Sort samples along that feature; by
        # copying the values into an array and
        # sorting the array in a manner which utilizes the cache more
        # effectively.
        if (
            missing_values_in_feature_mask is not None
            and missing_values_in_feature_mask[current_feature]
        ):
            i, current_end = self.start, self.end - 1
            # Missing values are placed at the end and do not participate in the sorting.
            while i <= current_end:
                # Finds the right-most value that is not missing so that
                # it can be swapped with missing values at its left.
                if ivy.isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue

                # X[samples[current_end], current_feature] is a non-missing value
                if ivy.isnan(X[samples[i], current_feature]):
                    samples[i], samples[current_end] = samples[current_end], samples[i]
                    n_missing += 1
                    current_end -= 1

                feature_values[i] = X[samples[i], current_feature]
                i += 1
        else:
            # When there are no missing values, we only need to copy the data into
            # feature_values
            for i in range(self.start, self.end):
                feature_values[i] = X[int(samples[i]), int(current_feature)]
        
        sort(
            feature_values[self.start:self.end],
            samples[self.start:self.end],
            self.end - self.start - n_missing,
        )
        self.n_missing = n_missing


class SparsePartitioner:
    """
    Partitioner specialized for sparse CSC data.

    Note that this partitioner is agnostic to the splitting strategy
    (best vs. random).
    """

    samples = []
    feature_values = []
    start = 0
    end = 0
    n_missing = 0
    missing_values_in_feature_mask = []

    X_data = []
    X_indices = []
    X_indptr = []

    n_total_samples = 0

    index_to_samples = []
    sorted_samples = []

    start_positive = 0
    end_negative = 0
    is_samples_sorted = False

    def __init__(
        self,
        X,
        samples: list,
        n_samples: int,
        feature_values: list,
        missing_values_in_feature_mask: list,
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
        self.index_to_samples = ivy.full(
            n_total_samples, fill_value=-1, dtype=ivy.int32
        )
        self.sorted_samples = ivy.empty(n_samples, dtype=ivy.int32)

        p = 0
        for p in range(n_samples):
            self.index_to_samples[samples[p]] = p

        self.missing_values_in_feature_mask = missing_values_in_feature_mask

    def init_node_split(self, start: int, end: int):
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.is_samples_sorted = 0
        self.n_missing = 0

    def sort_samples_and_feature_values(self, current_feature: int):
        # Simultaneously sort based on feature_values
        feature_values = self.feature_values
        index_to_samples = self.index_to_samples
        samples = self.samples

        self.extract_nnz(current_feature)
        # Sort the positive and negative parts of `feature_values`
        sort(
            feature_values[self.start],
            samples[self.start],
            self.end_negative - self.start,
        )
        if self.start_positive < self.end:
            sort(
                feature_values[self.start_positive],
                samples[self.start_positive],
                self.end - self.start_positive,
            )

        # Update index_to_samples to take into account the sort
        for p in range(self.start, self.end_negative):
            index_to_samples[samples[p]] = p
        for p in range(self.start_positive, self.end):
            index_to_samples[samples[p]] = p

        # Add one or two zeros in feature_values, if there is any
        if self.end_negative < self.start_positive:
            self.start_positive -= 1
            feature_values[self.start_positive] = 0.0

            if self.end_negative != self.start_positive:
                feature_values[self.end_negative] = 0.0
                self.end_negative += 1

        # XXX: When sparse supports missing values, this should be set to the
        # number of missing values for current_feature
        self.n_missing = 0

    def find_min_max(
        self,
        current_feature: int,
        min_feature_value_out: list,
        max_feature_value_out: list,
    ):
        # Find the minimum and maximum value for current_feature
        p = 0
        min_feature_value = 0
        max_feature_value = 0
        feature_values = self.feature_values

        self.extract_nnz(current_feature)

        if self.end_negative != self.start_positive:
            # There is a zero
            min_feature_value = 0
            max_feature_value = 0
        else:
            min_feature_value = feature_values[self.start]
            max_feature_value = min_feature_value

        # Find min, max in feature_values[start:end_negative]
        for p in range(self.start, self.end_negative):
            current_feature_value = feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        # Update min, max give feature_values[start_positive:end]
        for p in range(self.start_positive, self.end):
            current_feature_value = feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        min_feature_value_out[0] = min_feature_value
        max_feature_value_out[0] = max_feature_value

    def next_p(self, p_prev: list, p: list):
        # Compute the next p_prev and p for iterating over feature values
        p_next = 0
        feature_values = self.feature_values

        if p[0] + 1 != self.end_negative:
            p_next = p[0] + 1
        else:
            p_next = self.start_positive

        while (
            p_next < self.end
            and feature_values[p_next] <= feature_values[p[0]] + FEATURE_THRESHOLD
        ):
            p[0] = p_next
            if p[0] + 1 != self.end_negative:
                p_next = p[0] + 1
            else:
                p_next = self.start_positive

        p_prev[0] = p[0]
        p[0] = p_next

    def partition_samples(self, current_threshold: float):
        # Partition samples for feature_values at the current_threshold
        return self._partition(current_threshold, self.start_positive)

    def partition_samples_final(
        self, best_pos: int, best_threshold: float, best_feature: int, n_missing: int
    ):
        # Partition samples for X at the best_threshold and best_feature
        self.extract_nnz(best_feature)
        self._partition(best_threshold, best_pos)

    def _partition(self, threshold: float, zero_pos: int):
        # Partition samples based on threshold
        p = 0
        partition_end = 0
        index_to_samples = self.index_to_samples
        feature_values = self.feature_values
        samples = self.samples

        if threshold < 0.0:
            p = self.start
            partition_end = self.end_negative
        elif threshold > 0.0:
            p = self.start_positive
            partition_end = self.end
        else:
            # Data are already split
            return zero_pos

        while p < partition_end:
            if feature_values[p] <= threshold:
                p += 1

            else:
                partition_end -= 1

                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end],
                    feature_values[p],
                )
                sparse_swap(index_to_samples, samples, p, partition_end)

        return partition_end

    def extract_nnz(self, feature):
        """
        Extract and partition values for a given feature.

        The extracted values are partitioned between negative values
        feature_values[start:end_negative[0]] and positive values
        feature_values[start_positive[0]:end].
        The samples and index_to_samples are modified according to this
        partition.

        The extraction corresponds to the intersection between the arrays
        X_indices[indptr_start:indptr_end] and samples[start:end].
        This is done efficiently using either an index_to_samples based approach
        or binary search based approach.

        Parameters
        ----------
        feature : ,
            Index of the feature we want to extract non zero value.
        """
        samples = self.samples
        feature_values = self.feature_values
        indptr_start = (self.X_indptr[feature],)
        indptr_end = self.X_indptr[feature + 1]
        n_indices = int(indptr_end - indptr_start)
        n_samples = self.end - self.start
        index_to_samples = self.index_to_samples
        sorted_samples = self.sorted_samples
        X_indices = self.X_indices
        X_data = self.X_data

        # Use binary search if n_samples * log(n_indices) <
        # n_indices and index_to_samples approach otherwise.
        # O(n_samples * log(n_indices)) is the running time of binary
        # search and O(n_indices) is the running time of index_to_samples
        # approach.
        if (1 - self.is_samples_sorted) * n_samples * ivy.log(
            n_samples
        ) + n_samples * ivy.log(n_indices) < EXTRACT_NNZ_SWITCH * n_indices:
            extract_nnz_binary_search(
                X_indices,
                X_data,
                indptr_start,
                indptr_end,
                samples,
                self.start,
                self.end,
                index_to_samples,
                feature_values,
                self.end_negative,
                self.start_positive,
                sorted_samples,
                self.is_samples_sorted,
            )

        # Using an index to samples  technique to extract non zero values
        # index_to_samples is a mapping from X_indices to samples
        else:
            extract_nnz_index_to_samples(
                X_indices,
                X_data,
                indptr_start,
                indptr_end,
                samples,
                self.start,
                self.end,
                index_to_samples,
                feature_values,
                self.end_negative,
                self.start_positive,
            )


def binary_search(
    sorted_array: list, start: int, end: int, value: int, index: list, new_start: list
):
    """
    Return the index of value in the sorted array.

    If not found, return -1. new_start is the last pivot + 1
    """
    pivot = 0
    index[0] = -1
    while start < end:
        pivot = start + (end - start) / 2

        if sorted_array[pivot] == value:
            index[0] = pivot
            start = pivot + 1
            break

        if sorted_array[pivot] < value:
            start = pivot + 1
        else:
            end = pivot
    new_start[0] = start


def extract_nnz_index_to_samples(
    X_indices: list,
    X_data: list,
    indptr_start: int,
    indptr_end: int,
    samples: list,
    start: int,
    end: int,
    index_to_samples: list,
    feature_values: list,
    end_negative: list,
    start_positive: list,
):
    """
    Extract and partition values for a feature using index_to_samples.

    Complexity is O(indptr_end - indptr_start).
    """
    k = 0
    index = 0
    end_negative_ = start
    start_positive_ = end

    for k in range(indptr_start, indptr_end):
        if start <= index_to_samples[X_indices[k]] < end:
            if X_data[k] > 0:
                start_positive_ -= 1
                feature_values[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                feature_values[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


def extract_nnz_binary_search(
    X_indices: list,
    X_data: list,
    indptr_start: int,
    indptr_end: int,
    samples: list,
    start: int,
    end: int,
    index_to_samples: list,
    feature_values: list,
    end_negative: list,
    start_positive: list,
    sorted_samples: list,
    is_samples_sorted: list,
):
    """
    Extract and partition values for a given feature using binary search.

    If n_samples = end - start and n_indices = indptr_end - indptr_start,
    the complexity is

        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
          n_samples * log(n_indices)).
    """
    n_samples = 0

    if not is_samples_sorted[0]:
        n_samples = end - start
        sorted_samples[start : n_samples * 4] = samples[start : n_samples * 4]
        sorted_samples.sort()
        is_samples_sorted[0] = 1

    while indptr_start < indptr_end and sorted_samples[start] > X_indices[indptr_start]:
        indptr_start += 1

    while (
        indptr_start < indptr_end
        and sorted_samples[end - 1] < X_indices[indptr_end - 1]
    ):
        indptr_end -= 1

    p = start
    index = 0
    k = 0
    end_negative_ = start
    start_positive_ = end

    while p < end and indptr_start < indptr_end:
        # Find index of sorted_samples[p] in X_indices
        binary_search(
            X_indices, indptr_start, indptr_end, sorted_samples[p], k, indptr_start
        )

        if k != -1:
            # If k != -1, we have found a non zero value

            if X_data[k] > 0:
                start_positive_ -= 1
                feature_values[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                feature_values[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1
        p += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


def sparse_swap(index_to_samples: list, samples: list, pos_1: int, pos_2: int):
    """Swap sample pos_1 and pos_2 preserving sparse invariant."""
    samples[pos_1], samples[pos_2] = samples[pos_2], samples[pos_1]
    index_to_samples[samples[pos_1]] = pos_1
    index_to_samples[samples[pos_2]] = pos_2


class BestSplitter(Splitter):
    """Splitter for finding the best split on dense data."""

    def init(
        self,
        X,
        y: list,
        sample_weight: list,
        missing_values_in_feature_mask: list,
        *args
    ):
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask, *args)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    def node_split(
        self, impurity: float, split: SplitRecord, n_constant_features: list
    ):
        _, n_constant_features = node_split_best(
            self,
            self.partitioner,
            self.criterion,
            impurity,
            split,
            n_constant_features,
        )
        return _


class BestSparseSplitter(Splitter):
    """Splitter for finding the best split, using sparse data."""

    def init(
        self, X, y: list, sample_weight: list, missing_values_in_feature_mask: list
    ):
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X,
            self.samples,
            self.n_samples,
            self.feature_values,
            missing_values_in_feature_mask,
        )

    def node_split(
        self, impurity: float, split: SplitRecord, n_constant_features: list
    ):
        _, n_constant_features = node_split_best(
            self,
            self.partitioner,
            self.criterion,
            impurity,
            split,
            n_constant_features,
        )
        return _


class RandomSplitter(Splitter):
    """Splitter for finding the best random split on dense data."""

    def init(
        self,
        X,
        y: list,
        sample_weight: list,
        missing_values_in_feature_mask: list,
    ):
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    def node_split(self, impurity, split, n_constant_features):
        _, n_constant_features = node_split_random(
            self,
            self.partitioner,
            self.criterion,
            impurity,
            split,
            n_constant_features,
        )
        return _


class RandomSparseSplitter(Splitter):
    """Splitter for finding the best random split, using sparse data."""

    def init(
        self,
        X,
        y: list,
        sample_weight: list,
        missing_values_in_feature_mask: list,
    ):
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X,
            self.samples,
            self.n_samples,
            self.feature_values,
            missing_values_in_feature_mask,
        )

    def node_split(
        self, impurity: float, split: SplitRecord, n_constant_features: list
    ):
        _, n_constant_features = node_split_random(
            self,
            self.partitioner,
            self.criterion,
            impurity,
            split,
            n_constant_features,
        )
        return _
