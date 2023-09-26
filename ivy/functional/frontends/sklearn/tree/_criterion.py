import ivy
from ._utils import WeightedMedianCalculator
import numpy as np


EPSILON = 10 * np.finfo("double").eps


class Criterion:
    """
    Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is
    using different metrics.
    """

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def init(
        self,
        y,
        sample_weight,
        weighted_n_samples: float,
        sample_indices: list,
        start: int,
        end: int,
    ):
        """
        Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
            stored as a Cython memoryview.
        sample_weight : ndarray, dtype=DOUBLE_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : double
            The total weight of the samples being considered
        sample_indices : ndarray, dtype=SIZE_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node
        """
        pass

    def init_missing(self, n_missing):
        """
        Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]

        Parameters
        ----------
        n_missing: SIZE_t
            Number of missing values for specific feature.
        """
        pass

    def reset(self):
        """
        Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """
        pass

    def reverse_reset(self):
        """
        Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    def update(self, new_pos):
        """
        Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        This updates the collected statistics by moving sample_indices[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the sample_indices in the right child
        """
        pass

    def node_impurity(self):
        """
        Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of the
        current node, i.e. the impurity of sample_indices[start:end].
        This is the primary function of the criterion class. The smaller
        the impurity the better.
        """
        pass

    def children_impurity(self, impurity_left, impurity_right):
        """
        Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of sample_indices[start:pos] + the impurity
        of sample_indices[pos:end].

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """
        return impurity_left, impurity_right

    def node_value(self, dest):
        """
        Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of sample_indices[start:end] and save the value into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """
        return dest

    def proxy_impurity_improvement(self):
        """
        Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this
        value also maximizes the impurity improvement. It neglects all
        constant terms of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        impurity_left = 0.0
        impurity_right = 0.0
        impurity_left, impurity_right = self.children_impurity(
            impurity_left, impurity_right
        )

        return (
            -self.weighted_n_right * impurity_right
            - self.weighted_n_left * impurity_left
        )

    def impurity_improvement(
        self, impurity_parent: float, impurity_left: float, impurity_right: float
    ):
        """
        Compute the improvement in impurity.

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child.

        Parameters
        ----------
        impurity_parent : double
            The initial impurity of the parent node before the split

        impurity_left : double
            The impurity of the left child

        impurity_right : double
            The impurity of the right child

        Return
        ------
        double : improvement in impurity after the split occurs
        """
        return (self.weighted_n_node_samples / self.weighted_n_samples) * (
            impurity_parent
            - (self.weighted_n_right / self.weighted_n_node_samples) * impurity_right
            - (self.weighted_n_left / self.weighted_n_node_samples) * impurity_left
        )

    def init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        pass


class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    def __init__(self, n_outputs: int, n_classes: ivy.Array):
        """
        Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """
        self.start = 0
        self.pos = 0
        self.end = 0
        self.missing_go_to_left = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        self.n_classes = ivy.empty(n_outputs, dtype=ivy.int16)

        k = 0
        max_n_classes = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > max_n_classes:
                max_n_classes = n_classes[k]

        self.max_n_classes = max_n_classes

        # Count labels for each output
        self.sum_total = ivy.zeros((n_outputs, max_n_classes), dtype=ivy.float64)
        self.sum_left = ivy.zeros((n_outputs, max_n_classes), dtype=ivy.float64)
        self.sum_right = ivy.zeros((n_outputs, max_n_classes), dtype=ivy.float64)

    def __reduce__(self):
        return (
            type(self),
            (self.n_outputs, ivy.asarray(self.n_classes)),
            self.__getstate__(),
        )

    def init(
        self,
        y: list,
        sample_weight: list,
        weighted_n_samples: float,
        sample_indices: int,
        start: int,
        end: int,
    ):
        """
        Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency.
        sample_weight : ndarray, dtype=DOUBLE_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : double
            The total weight of all samples
        sample_indices : ndarray, dtype=SIZE_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        i = 0
        p = 0
        k = 0
        c = 0
        w = 1.0

        for k in range(self.n_outputs):
            self.sum_total[k, 0 : int(self.n_classes[k]) * 8] = 0

        for p in range(start, end):
            # print(f"{p=}")
            i = sample_indices[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0.
            if sample_weight is not None:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = int(self.y[i, k])
                self.sum_total[k, c] += w

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    def init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        self.sum_missing = ivy.zeros(
            (self.n_outputs, self.max_n_classes), dtype=ivy.float64
        )

    def init_missing(self, n_missing):
        """
        Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        i = 0
        p = 0
        k = 0
        c = 0
        w = 1.0

        self.n_missing = n_missing
        if n_missing == 0:
            return

        self.sum_missing[0, 0 : self.max_n_classes * self.n_outputs * 8] = 0
        self.weighted_n_missing = 0.0

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                c = int(self.y[i, k])
                self.sum_missing[k, c] += w

            self.weighted_n_missing += w

    def reset(self):
        """
        Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        self.pos = self.start
        self.weighted_n_left, self.weighted_n_right = _move_sums_classification(
            self,
            self.sum_left,
            self.sum_right,
            self.weighted_n_left,
            self.weighted_n_right,
            self.missing_go_to_left,
        )
        return 0

    def reverse_reset(self):
        """
        Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        self.pos = self.end
        self.weighted_n_right, self.weighted_n_left = _move_sums_classification(
            self,
            self.sum_right,
            self.sum_left,
            self.weighted_n_right,
            self.weighted_n_left,
            not self.missing_go_to_left,
        )
        return 0

    def update(self, new_pos):
        """
        Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        pos = self.pos
        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        end_non_missing = self.end - self.n_missing

        sample_indices = self.sample_indices
        sample_weight = self.sample_weight

        i = 0
        p = 0
        k = 0
        c = 0
        w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    c = int(self.y[i, k])
                    self.sum_left[k, c] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    c = int(self.y[i, k])
                    self.sum_left[k, c] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left

        for k in range(self.n_outputs):
            for c in range(self.max_n_classes):
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]

        self.pos = new_pos
        return 0

    def node_impurity(self):
        pass

    def children_impurity(self, impurity_left: float, impurity_right: float):
        pass

    def node_value(self, dest):
        """
        Compute the node value of sample_indices[start:end] and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        for k in range(self.n_outputs):
            dest[: self.n_classes[k] * 8] = self.sum_total[k, 0 : self.n_classes[k] * 8]
            dest += self.max_n_classes

        return dest


class RegressionCriterion(Criterion):
    """
    Abstract regression criterion for regression problems.

    This handles cases where the target is a continuous value and is
    evaluated by computing the variance of the target values left and
    right of the split point.
    """

    def __init__(self, n_outputs: int, n_samples: int):
        """
        Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : int
            The number of targets to be predicted

        n_samples : int
            The total number of samples to fit on
        """
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        self.sq_sum_total = 0.0

        self.sum_total = ivy.zeros(n_outputs, dtype=ivy.float64)
        self.sum_left = ivy.zeros(n_outputs, dtype=ivy.float64)
        self.sum_right = ivy.zeros(n_outputs, dtype=ivy.float64)

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    def init(
        self,
        y,
        sample_weight,
        weighted_n_samples: float,
        sample_indices: int,
        start: int,
        end: int,
    ):
        """
        Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end]
        and children sample_indices[start:start] and
        sample_indices[start:end].
        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        i = 0
        p = 0
        k = 0
        y_ik = 0.0
        w_y_ik = 0.0
        w = 1.0
        self.sq_sum_total = 0.0

        self.sum_total[0 : self.n_outputs * 8] = 0.0

        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    def init_sum_missing(self):
        """Initialize sum_missing to hold sums for missing values."""
        self.sum_missing = ivy.zeros(self.n_outputs, dtype=ivy.float64)

    def init_missing(self, n_missing: int):
        """
        Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        i = 0
        p = 0
        k = 0
        y_ik = 0.0
        w_y_ik = 0.0
        w = 1.0

        self.n_missing = n_missing
        if n_missing == 0:
            return

        self.sum_missing[0 : self.n_outputs * 8] = 0.0

        self.weighted_n_missing = 0.0

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            w = self.sample_weight[i] if self.sample_weight is not None else 1.0

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_missing[k] += w_y_ik

            self.weighted_n_missing += w

    def reset(self):
        """Reset the criterion at pos=start."""
        self.pos = self.start
        self.weighted_n_left, self.weighted_n_right = self._move_sums_regression(
            self.sum_left,
            self.sum_right,
            self.weighted_n_left,
            self.weighted_n_right,
            self.missing_go_to_left,
        )
        return 0

    def reverse_reset(self):
        """Reset the criterion at pos=end."""
        self.pos = self.end
        self.weighted_n_right, self.weighted_n_left = self._move_sums_regression(
            self.sum_right,
            self.sum_left,
            self.weighted_n_right,
            self.weighted_n_left,
            not self.missing_go_to_left,
        )
        return 0

    def update(self, new_pos: int):
        """Update statistics by moving sample_indices[pos:new_pos] to the left."""
        sample_weight = self.sample_weight
        sample_indices = self.sample_indices

        pos = self.pos

        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        end_non_missing = self.end - self.n_missing
        i = 0
        p = 0
        k = 0
        w = 1.0

        for k in range(self.n_outputs):
            self.sum_left[k] = 0.0

        w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k] += w * self.y[i, k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k] -= w * self.y[i, k]

                self.weighted_n_left -= w

        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left

        for k in range(self.n_outputs):
            self.sum_right[k] = self.sum_total[k] - self.sum_left[k]

        self.pos = new_pos
        return 0

    def node_impurity(self):
        pass

    def children_impurity(self, impurity_left: float, impurity_right: float):
        return impurity_left, impurity_right

    def node_value(self, dest: float):
        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples

        return dest


class Entropy(ClassificationCriterion):
    """
    Cross Entropy impurity criterion for classification.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm * sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -sum_{k=0}^{K-1} count_k * log(count_k)
    """

    def node_impurity(self):
        """
        Evaluate the impurity of the current node.

        Evaluate the cross-entropy criterion as impurity of the current
        node, i.e., the impurity of sample_indices[start:end]. The
        smaller the impurity the better.
        """
        entropy = 0.0
        count_k = 0.0
        k = 0
        c = 0

        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                count_k = self.sum_total[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * ivy.log(count_k)

        return entropy / self.n_outputs

    def children_impurity(self, impurity_left: float, impurity_right: float):
        """
        Evaluate the impurity in children nodes.

        i.e., the impurity of the left child (sample_indices[start:pos]) and the
        impurity of the right child (sample_indices[pos:end]).

        Returns
        -------
        impurity_left : list
            The impurity of the left child
        impurity_right : list
            The impurity of the right child
        """
        entropy_left = 0.0
        entropy_right = 0.0
        count_k = 0.0
        k = 0
        c = 0

        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                count_k = self.sum_left[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * ivy.log(count_k)

                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * ivy.log(count_k)

        impurity_left = entropy_left / self.n_outputs
        impurity_right = entropy_right / self.n_outputs

        return impurity_left, impurity_right


class Gini(ClassificationCriterion):
    """
    Gini Index impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """

    def node_impurity(self):
        """
        Evaluate the impurity of the current node.

        Evaluate the Gini criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the
        impurity the better.
        """
        gini = 0.0
        sq_count = 0.0
        count_k = 0.0
        k = 0
        c = 0

        for k in range(self.n_outputs):
            sq_count = 0.0

            for c in range(int(self.n_classes[k])):
                count_k = self.sum_total[k, c]
                sq_count += count_k * count_k

            gini += 1.0 - sq_count / (
                self.weighted_n_node_samples * self.weighted_n_node_samples
            )

        return gini / self.n_outputs

    def children_impurity(
        self,
        impurity_left: float,
        impurity_right: float,
    ):
        """
        Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node to
        impurity_right : double pointer
            The memory address to save the impurity of the right node to
        """
        gini_left = 0.0
        gini_right = 0.0
        sq_count_left = 0.0
        sq_count_right = 0.0
        count_k = 0.0
        k = 0
        c = 0

        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0

            for c in range(self.n_classes[k]):
                count_k = self.sum_left[k, c]
                sq_count_left += count_k * count_k

                count_k = self.sum_right[k, c]
                sq_count_right += count_k * count_k

            gini_left += 1.0 - sq_count_left / (
                self.weighted_n_left * self.weighted_n_left
            )
            gini_right += 1.0 - sq_count_right / (
                self.weighted_n_right * self.weighted_n_right
            )

        impurity_left = gini_left / self.n_outputs
        impurity_right = gini_right / self.n_outputs

        return impurity_left, impurity_right


class MSE(RegressionCriterion):
    """
    Mean squared error impurity criterion.

    MSE = var_left + var_right
    """

    def node_impurity(self):
        """
        Evaluate the impurity of the current node.

        Evaluate the MSE criterion as impurity of the current node, i.e.
        the impurity of sample_indices[start:end]. The smaller the
        impurity the better.
        """
        impurity = 0.0
        k = 0

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (self.sum_total[k] / self.weighted_n_node_samples) ** 2.0

        return impurity / self.n_outputs

    def proxy_impurity_improvement(self):
        """
        Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        The MSE proxy is derived from

            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2

        Neglecting constant terms, this gives:

            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
        """
        k = 0
        proxy_impurity_left = 0.0
        proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
            proxy_impurity_right += self.sum_right[k] * self.sum_right[k]

        return (
            proxy_impurity_left / self.weighted_n_left
            + proxy_impurity_right / self.weighted_n_right
        )

    def children_impurity(self, impurity_left: float, impurity_right: float):
        """
        Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos])
        and the impurity the right child (sample_indices[pos:end]).
        """
        sample_weight = self.sample_weight
        sample_indices = self.sample_indices
        pos = self.pos
        start = self.start

        y_ik = 0.0

        sq_sum_left = 0.0
        sq_sum_right = 0.0

        for p in range(start, pos):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left = sq_sum_left / self.weighted_n_left
        impurity_right = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right -= (self.sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left /= self.n_outputs
        impurity_right /= self.n_outputs

        return impurity_left, impurity_right


class MAE(RegressionCriterion):
    """
    Mean absolute error impurity criterion.

    MAE = (1 / n) * (sum_i |y_i - f_i|), where y_i is the true
    value and f_i is the predicted value.
    """

    def __init__(self, n_outputs, n_samples):
        """
        Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on
        """
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.node_medians = ivy.zeros(n_outputs, dtype=ivy.float64)

        self.left_child = ivy.empty(n_outputs, dtype="object")
        self.right_child = ivy.empty(n_outputs, dtype="object")
        # initialize WeightedMedianCalculators
        for k in range(n_outputs):
            self.left_child[k] = WeightedMedianCalculator(n_samples)
            self.right_child[k] = WeightedMedianCalculator(n_samples)

    def init(
        self,
        y: list,
        sample_weight: list,
        weighted_n_samples: float,
        sample_indices: list,
        start: int,
        end: int,
    ):
        """
        Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end]
        and children sample_indices[start:start] and
        sample_indices[start:end].
        """
        i = 0
        p = 0
        k = 0
        w = 1.0

        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        left_child = self.left_child
        right_child = self.right_child

        for k in range(self.n_outputs):
            left_child[k].reset()
            right_child[k].reset()

        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                # push method ends up calling safe_realloc, hence `except -1`
                # push all values to the right side,
                # since pos = start initially anyway
                right_child[k].push(self.y[i, k], w)

            self.weighted_n_node_samples += w

        # calculate the node medians
        for k in range(self.n_outputs):
            self.node_medians[k] = right_child[k].get_median()

        # Reset to pos=start
        self.reset()
        return 0

    def init_missing(self, n_missing):
        """Raise error if n_missing != 0."""
        if n_missing == 0:
            return
        raise ValueError("Missing values are not supported for MAE.")

    def reset(self):
        """
        Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        i = 0
        k = 0
        value = 0.0
        weight = 0.0

        left_child = self.left_child
        right_child = self.right_child

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

        # reset the WeightedMedianCalculators, left should have no
        # elements and right should have all elements.

        for k in range(self.n_outputs):
            # if left has no elements, it's already reset
            for i in range(left_child[k].size()):
                # remove everything from left and put it into right
                value, weight = left_child[k].pop()
                right_child[k].push(value, weight)
        return 0

    def reverse_reset(self):
        """
        Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

        value = 0.0
        weight = 0.0
        left_child = self.left_child
        right_child = self.right_child

        # reverse reset the WeightedMedianCalculators, right should have no
        # elements and left should have all elements.
        for k in range(self.n_outputs):
            # if right has no elements, it's already reset
            for i in range(right_child[k].size()):
                # remove everything from right and put it into left
                value, weight = right_child[k].pop()
                # push method ends up calling safe_realloc, hence `except -1`
                left_child[k].push(value, weight)
        return 0

    def update(self, new_pos):
        """
        Updated statistics by moving sample_indices[pos:new_pos] to the left.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        sample_weight = self.sample_weight
        sample_indices = self.sample_indices

        left_child = self.left_child
        right_child = self.right_child

        pos = self.pos
        end = self.end
        i = 0
        p = 0
        k = 0
        w = 1.0

        # Update statistics up to new_pos
        #
        # We are going to update right_child and left_child
        # from the direction that require the least amount of
        # computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                # remove y_ik and its weight w from right and add to left
                right_child[k].remove(self.y[i, k], w)
                # push method ends up calling safe_realloc, hence except -1
                left_child[k].push(self.y[i, k], w)

            self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                # remove y_ik and its weight w from left and add to right
                left_child[k].remove(self.y[i, k], w)
                right_child[k].push(self.y[i, k], w)

            self.weighted_n_left -= w

        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.pos = new_pos
        return 0

    def node_value(self, dest: float):
        """Computes the node value of sample_indices[start:end] into dest."""
        for k in range(self.n_outputs):
            dest[k] = self.node_medians[k]

        return dest

    def node_impurity(self):
        """
        Evaluate the impurity of the current node.

        Evaluate the MAE criterion as impurity of the current node, i.e.
        the impurity of sample_indices[start:end]. The smaller the
        impurity the better.
        """
        sample_weight = self.sample_weight
        sample_indices = self.sample_indices
        i = 0
        p = 0
        k = 0
        w = 1.0
        impurity = 0.0

        for k in range(self.n_outputs):
            for p in range(self.start, self.end):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]
                else:
                    w = 1.0

                impurity += abs(self.y[i, k] - self.node_medians[k]) * w

        return impurity / (self.weighted_n_node_samples * self.n_outputs)

    def children_impurity(self, p_impurity_left: float, p_impurity_right: float):
        """
        Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos])
        and the impurity the right child (sample_indices[pos:end]).
        """
        sample_weight = self.sample_weight
        sample_indices = self.sample_indices

        start = self.start
        pos = self.pos
        end = self.end

        i = 0
        p = 0
        k = 0
        median = 0.0
        w = 1.0
        impurity_left = 0.0
        impurity_right = 0.0

        left_child = self.left_child
        right_child = self.right_child

        for k in range(self.n_outputs):
            median = left_child[k]
            for p in range(start, pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                impurity_left += ivy.abs(self.y[i, k] - median) * w
        p_impurity_left = impurity_left / (self.weighted_n_left * self.n_outputs)

        for k in range(self.n_outputs):
            median = right_child[k].get_median()
            for p in range(pos, end):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                impurity_right += ivy.abs(self.y[i, k] - median) * w
        p_impurity_right = impurity_right / (self.weighted_n_right * self.n_outputs)

        return p_impurity_left, p_impurity_right


class Poisson(RegressionCriterion):
    def __init__(self, n_outputs):
        self.n_outputs = n_outputs

    def node_impurity(self):
        return self.poisson_loss(
            self.start, self.end, self.sum_total, self.weighted_n_node_samples
        )

    def proxy_impurity_improvement(self):
        proxy_impurity_left = 0.0
        proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            if (self.sum_left[k] <= EPSILON) or (self.sum_right[k] <= EPSILON):
                return -ivy.inf
            else:
                y_mean_left = self.sum_left[k] / self.weighted_n_left
                y_mean_right = self.sum_right[k] / self.weighted_n_right
                proxy_impurity_left -= self.sum_left[k] * ivy.log(y_mean_left)
                proxy_impurity_right -= self.sum_right[k] * ivy.log(y_mean_right)

        return -proxy_impurity_left - proxy_impurity_right

    def children_impurity(self, impurity_left: float, impurity_right: float):
        start = self.start
        pos = self.pos
        end = self.end

        impurity_left = self.poisson_loss(
            start, pos, self.sum_left, self.weighted_n_left
        )
        impurity_right = self.poisson_loss(
            pos, end, self.sum_right, self.weighted_n_right
        )

        return impurity_left, impurity_right

    def poisson_loss(self, start, end, y_sum, weight_sum):
        y_mean = 0.0
        poisson_loss = 0.0
        w = 1.0

        for k in range(self.n_outputs):
            if y_sum[k] <= EPSILON:
                return ivy.inf

            y_mean = y_sum[k] / weight_sum

            for p in range(start, end):
                i = self.sample_indices[p]

                if self.sample_weight is not None:
                    w = self.sample_weight[i]

                poisson_loss += w * self.y[i, k] * ivy.log(self.y[i, k] / y_mean)

        return poisson_loss / (weight_sum * self.n_outputs)


class FriedmanMSE(MSE):
    """
    Mean squared error impurity criterion with improvement score by Friedman.

    Uses the formula (35) in Friedman's original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    def proxy_impurity_improvement(self):
        """
        Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this
        value also maximizes the impurity improvement. It neglects all
        constant terms of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        total_sum_left = 0.0
        total_sum_right = 0.0

        for k in range(self.n_outputs):
            total_sum_left += self.sum_left[k]
            total_sum_right += self.sum_right[k]

        diff = (
            self.weighted_n_right * total_sum_left
            - self.weighted_n_left * total_sum_right
        )

        return diff * diff / (self.weighted_n_left * self.weighted_n_right)

    def impurity_improvement(self, impurity_parent, impurity_left, impurity_right):
        total_sum_left = 0.0
        total_sum_right = 0.0

        for k in range(self.n_outputs):
            total_sum_left += self.sum_left[k]
            total_sum_right += self.sum_right[k]

        diff = (
            self.weighted_n_right * total_sum_left
            - self.weighted_n_left * total_sum_right
        ) / self.n_outputs

        return (
            diff
            * diff
            / (
                self.weighted_n_left
                * self.weighted_n_right
                * self.weighted_n_node_samples
            )
        )


# --- Helpers --- #
# --------------- #


def _move_sums_classification(
    criterion, sum_1, sum_2, weighted_n_1, weighted_n_2, put_missing_in_1
):
    """
    Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values go to sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    # if criterion.n_missing != 0 and put_missing_in_1:
    if put_missing_in_1:
        for k in range(criterion.n_outputs):
            n_bytes = criterion.n_classes[k] * 8
            sum_1[k, 0:n_bytes] = criterion.sum_missing[k, 0:n_bytes]

        for k in range(criterion.n_outputs):
            for c in range(criterion.n_classes[k]):
                sum_2[k, c] = criterion.sum_total[k, c] - criterion.sum_missing[k, c]

        weighted_n_1 = criterion.weighted_n_missing
        weighted_n_2 = criterion.weighted_n_node_samples - criterion.weighted_n_missing
    else:
        # Assigning sum_2 = sum_total for all outputs.
        for k in range(criterion.n_outputs):
            n_bytes = int(criterion.n_classes[k]) * 8
            sum_1[k, 0:n_bytes] = 0
            sum_2[k, 0:n_bytes] = criterion.sum_total[k, 0:n_bytes]

        weighted_n_1 = 0.0
        weighted_n_2 = criterion.weighted_n_node_samples

    return weighted_n_1, weighted_n_2


def _move_sums_regression(
    criterion, sum_1, sum_2, weighted_n_1, weighted_n_2, put_missing_in_1
):
    """
    Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values go to sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    i = 0
    n_bytes = criterion.n_outputs * 8
    has_missing = criterion.n_missing != 0

    if has_missing and put_missing_in_1:
        sum_1[0:n_bytes] = criterion.sum_missing[0:n_bytes]
        for i in range(criterion.n_outputs):
            sum_2[i] = criterion.sum_total[i] - criterion.sum_missing[i]
        weighted_n_1[0] = criterion.weighted_n_missing
        weighted_n_2[0] = (
            criterion.weighted_n_node_samples - criterion.weighted_n_missing
        )
    else:
        sum_1[0:n_bytes] = 0
        # Assigning sum_2 = sum_total for all outputs.
        sum_2[0:n_bytes] = criterion.sum_total[0:n_bytes]
        weighted_n_1 = 0.0
        weighted_n_2 = criterion.weighted_n_node_samples

    return weighted_n_1, weighted_n_2
