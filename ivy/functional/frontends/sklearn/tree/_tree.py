import ivy

from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse import isspmatrix_csr

from ._splitter import SplitRecord, Splitter
import numpy as np

EPSILON = ivy.finfo(ivy.double).eps
INFINITY = ivy.inf
# Some handy constants (BestFirstTreeBuilder)
IS_FIRST = 1
IS_LEFT = 1
IS_NOT_FIRST = 0
IS_NOT_LEFT = 0


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature = None
        self.threshold = None
        self.impurity = None
        self.n_node_samples = None
        self.weighted_n_node_samples = None
        self.missing_go_to_left = None


class Tree:
    """
    Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of double, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """

    def __init__(self, n_features, n_classes, n_outputs):
        """Constructor."""
        self.n_features = None
        self.n_classes = None
        self.n_outputs = None
        self.max_n_classes = None
        self.max_depth = None
        self.node_count = None
        self.capacity = None
        self.nodes = []  # replaced it with array since this array will contain nodes
        self.value = None
        self.value_stride = None

        size_t_dtype = "int32"

        n_classes = _check_n_classes(n_classes, size_t_dtype)

        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = ivy.zeros(n_outputs, dtype=ivy.int32)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = None
        self.nodes = None

    def __del__(self):
        """Destructor."""
        # Free all inner structures
        self.n_classes = None
        self.value = None
        self.nodes = None

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        raise NotImplementedError

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        raise NotImplementedError

    def _resize(self, capacity):
        """
        Resize all inner arrays to `capacity`, if `capacity` == -1, then double the size
        of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            raise MemoryError()

    def _resize_c(self, capacity=float("inf")):
        """
        Guts of _resize.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes is None:
            return 0

        if capacity == INTPTR_MAX:
            if self.capacity == 0:
                capacity = 3
            else:
                capacity = 2 * self.capacity

        self.nodes = np.zeros(capacity, dtype="int32")
        self.value = np.zeros(capacity * self.value_stride, dtype="int32")

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            self.value[
                self.capacity * self.value_stride : capacity * self.value_stride
            ] = 0

        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    def _add_node(
        self,
        parent,
        is_left,
        is_leaf,
        feature,
        threshold,
        impurity,
        n_node_samples,
        weighted_n_node_samples,
        missing_go_to_left,
    ):
        """
        Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns -1 on error.
        """
        node_id = self.node_count

        node = (
            Node()
        )  # self.nodes contains a list of nodes, it returns the node at node_id location

        self.nodes.append(node)
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold
            node.missing_go_to_left = missing_go_to_left

        self.node_count += 1

        return node_id

    def predict(self, X):
        # Apply the model to the input data X
        predictions = self.apply(X)
        # Get the internal data as a NumPy array
        internal_data = self._get_value_ndarray()
        # Use the predictions to index the internal data
        out = internal_data[
            predictions
        ]  # not sure if this accurately translates to .take(self.apply(X), axis=0, mode='clip')
        # Reshape the output if the model is single-output
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    def apply(self, X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)

    def _apply_dense(self, X):
        if not isinstance(X, ivy.data_classes.array.array.Array):
            raise ValueError(
                "X should be a ivy.data_classes.array.array.Array, got %s" % type(X)
            )

        if X.dtype != "float32":
            raise ValueError("X.dtype should be float32, got %s" % X.dtype)

        X_tensor = X
        n_samples = X.shape[0]
        out = ivy.zeros(n_samples, dtype="float32")

        for i in range(n_samples):
            node = self.nodes[0]  # Start at the root node

            while node.left_child != _TREE_LEAF:
                X_i_node_feature = X_tensor[i, node.feature]

                if ivy.isnan(X_i_node_feature):
                    if node.missing_go_to_left:
                        node = self.nodes[node.left_child]
                    else:
                        node = self.nodes[node.right_child]
                elif X_i_node_feature <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]

            out[i] = self.nodes.index(node)  # Get the index of the terminal node

        return out

    # not so sure about sparse implimentation yet
    def _apply_sparse_csr(self, X):
        """Finds the terminal region (=leaf node) for each sample in sparse X."""
        if not isinstance(X, ivy.data_classes.array.array.Array):
            raise ValueError(
                "X should be a ivy.data_classes.array.array.Array, got %s" % type(X)
            )

        if X.dtype != "float32":
            raise ValueError("X.dtype should be float32, got %s" % X.dtype)

        n_samples, n_features = X.shape

        # Initialize output
        out = ivy.zeros(n_samples, dtype="float32")

        # Initialize auxiliary data structures on CPU
        feature_to_sample = ivy.full((n_features,), -1, dtype="float32")
        X_sample = ivy.zeros((n_features,), dtype="float32")

        for i in range(n_samples):
            node = self.nodes[0]  # Start from the root node

            for k in range(X.indptr[i], X.indptr[i + 1]):
                feature_to_sample[X.indices[k]] = i
                X_sample[X.indices[k]] = X.data[k]

            while node.left_child is not None:
                if feature_to_sample[node.feature] == i:
                    feature_value = X_sample[node.feature]
                else:
                    feature_value = ivy.array(
                        0, dtype="float32"
                    )  # feature value is computed during training

                threshold = ivy.array(node.threshold, dtype="float32")
                if feature_value <= threshold:
                    node = node.left_child
                else:
                    node = node.right_child

            # Get the index of the leaf node
            out[i] = self.nodes.index(node)

        return out

    def decision_path(self, X):
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    def _decision_path_dense(self, X):
        # Check input
        if not isinstance(X, ivy.data_classes.array.array.Array):
            raise ValueError(
                "X should be a ivy.data_classes.array.array.Array, got %s" % type(X)
            )

        if X.dtype != "float32":
            raise ValueError("X.dtype should be float32, got %s" % X.dtype)

        # Extract input
        n_samples = X.shape[0]

        # Initialize output
        indptr = ivy.zeros(n_samples + 1, dtype="float32")
        indices = ivy.zeros(n_samples * (1 + self.max_depth), dtype="float32")

        # Initialize auxiliary data-structure
        i = 0

        for i in range(n_samples):
            node = self.nodes[0]
            indptr[i + 1] = indptr[i]

            # Add all external nodes
            while node.left_child != _TREE_LEAF:
                indices[indptr[i + 1]] = node
                indptr[i + 1] += 1

                if X[i, node.feature] <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]

            # Add the leaf node
            indices[indptr[i + 1]] = node
            indptr[i + 1] += 1

        indices = indices[: indptr[n_samples]]
        data = ivy.ones(indices.shape, dtype="float32")
        # csr_matrix is not implemented
        out = csr_matrix((data, indices, indptr), shape=(n_samples, self.node_count))

        return out

    # not tested
    def _decision_path_sparse_csr(self, X):
        # Check input
        if not isspmatrix_csr(X):
            raise ValueError("X should be in csr_matrix format, got %s" % type(X))

        if X.dtype != "float32":
            raise ValueError("X.dtype should be float32, got %s" % X.dtype)

        # Extract input
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Initialize output
        indptr = ivy.zeros(n_samples + 1, dtype="int32")
        indices = ivy.zeros(n_samples * (1 + self.max_depth), dtype="int32")

        # Initialize auxiliary data-structure
        feature_value = 0.0
        node = None
        X_sample = ivy.zeros(n_features, dtype="float32")
        feature_to_sample = ivy.full(n_features, -1, dtype="int32")

        for i in range(n_samples):
            node = self.nodes
            indptr[i + 1] = indptr[i]

            for k in range(X_indptr[i], X_indptr[i + 1]):
                feature_to_sample[X_indices[k]] = i
                X_sample[X_indices[k]] = X_data[k]

            # While node not a leaf
            while node.left_child != _TREE_LEAF:
                indices[indptr[i + 1]] = node - self.nodes
                indptr[i + 1] += 1

                if feature_to_sample[node.feature] == i:
                    feature_value = X_sample[node.feature]
                else:
                    feature_value = 0.0

                if feature_value <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]

            # Add the leaf node
            indices[indptr[i + 1]] = node - self.nodes
            indptr[i + 1] += 1

        indices = indices[: indptr[n_samples]]
        data = ivy.ones(shape=len(indices), dtype="int32")
        out = csr_matrix((data, indices, indptr), shape=(n_samples, self.node_count))

        return out

    def compute_node_depths(self):
        depths = ivy.zeros(self.node_count, dtype="int32")
        children_left = self.children_left
        children_right = self.children_right
        node_count = self.node_count

        depths[0] = 1  # init root node
        for node_id in range(node_count):
            if children_left[node_id] != _TREE_LEAF:
                depth = depths[node_id] + 1
                depths[children_left[node_id]] = depth
                depths[children_right[node_id]] = depth

        return depths.base

    """
    This code is typically used after fitting a decision tree model to assess the importance of each feature in making predictions.
    The feature importances indicate the contribution of each feature to the overall decision-making process of the tree.
    """

    def compute_feature_importances(self, normalize=True):
        # Compute the importance of each feature (variable).

        # Create an array to store feature importances.
        importances = ivy.zeros(self.n_features)

        for node in self.nodes:
            if node.left_child is not None and node.right_child is not None:
                left = node.left_child
                right = node.right_child

                # Calculate the importance for the feature associated with this node.
                importance = (
                    node.weighted_n_node_samples * node.impurity
                    - left.weighted_n_node_samples * left.impurity
                    - right.weighted_n_node_samples * right.impurity
                )

                importances[node.feature] += importance

        if normalize:
            total_importance = ivy.sum(importances)

            if total_importance > 0.0:
                # Normalize feature importances to sum up to 1.0
                importances /= total_importance

        return importances

    def _get_value_ndarray(self):
        shape = (int(self.node_count), int(self.n_outputs), int(self.max_n_classes))
        arr = ivy.array(self.value, dtype="float32").reshape(shape)
        return arr

    """
    this code creates a NumPy structured array that wraps the internal nodes of a decision tree.
    This array can be used to access and manipulate the tree's nodes efficiently in Python.
    """

    def _get_node_tensor(self):
        # Create a tensor with a custom data type for the tree nodes
        nodes_tensor = ivy.zeros(self.node_count, dtype="float32")

        # Fill the tensor with node data
        for i, node in enumerate(self.nodes):
            nodes_tensor[i] = ivy.array(
                (
                    node.impurity,
                    node.n_node_samples,
                    node.weighted_n_node_samples,
                    node.left_child,
                    node.right_child,
                    node.feature,
                    node.threshold,
                    node.missing_go_to_left,
                ),
                dtype="float32",
            )

        # Attach a reference to the tree
        nodes_tensor.tree_reference = self

        return nodes_tensor

    """
    This code effectively computes the partial dependence values for a set of grid points based on a given decision tree.
    Partial dependence helps understand how the model's predictions change with variations in specific features while keeping other features constant.
    """

    def compute_partial_dependence(self, X, target_features, out):
        weight_stack = ivy.zeros(self.node_count, dtype="float32")
        node_idx_stack = ivy.zeros(self.node_count, dtype="int32")
        stack_size = 0

        for sample_idx in range(X.shape[0]):
            # Init stacks for the current sample
            stack_size = 1
            node_idx_stack[0] = 0  # Root node
            weight_stack[0] = 1.0  # All samples are in the root node

            while stack_size > 0:
                # Pop the stack
                stack_size -= 1
                current_node_idx = node_idx_stack[stack_size]
                current_node = self.nodes[current_node_idx]

                if current_node.left_child == -1:  # Leaf node
                    out[sample_idx] += (
                        weight_stack[stack_size] * self.value[current_node_idx]
                    )
                else:  # Non-leaf node
                    is_target_feature = current_node.feature in target_features
                    if is_target_feature:
                        # Push left or right child on the stack
                        if (
                            X[sample_idx, current_node.feature]
                            <= current_node.threshold
                        ):
                            node_idx_stack[stack_size] = current_node.left_child
                        else:
                            node_idx_stack[stack_size] = current_node.right_child
                        stack_size += 1
                    else:
                        # Push both children onto the stack and give a weight proportional to the number of samples going through each branch
                        left_child = current_node.left_child
                        right_child = current_node.right_child
                        left_sample_frac = (
                            self.nodes[left_child].weighted_n_node_samples
                            / current_node.weighted_n_node_samples
                        )
                        current_weight = weight_stack[stack_size]
                        weight_stack[stack_size] = current_weight * left_sample_frac
                        stack_size += 1

                        node_idx_stack[stack_size] = right_child
                        weight_stack[stack_size] = current_weight * (
                            1 - left_sample_frac
                        )
                        stack_size += 1

        # Sanity check. Should never happen.
        if not (0.999 < ivy.sum(weight_stack) < 1.001):
            raise ValueError(
                "Total weight should be 1.0 but was %.9f" % ivy.sum(weight_stack)
            )


# =============================================================================
# TreeBuilder
# =============================================================================


class TreeBuilder:
    """Interface for different tree building strategies."""

    def __init__(self):
        self.splitter = None
        self.min_samples_split = None
        self.min_samples_leaf = None
        self.min_weight_leaf = None
        self.max_depth = None
        self.min_impurity_decrease = None

    def build(
        self,
        tree: Tree,
        X,
        y,
        sample_weight=None,
        missing_values_in_feature_mask=None,
    ):
        """Build a decision tree from the training set (X, y)."""
        pass

    def _check_input(
        self,
        X,
        y,
        sample_weight,
    ):
        """Check input dtype, layout, and format."""
        if issparse(X):
            X = (
                X.tocsc()
            )  # tocsc() is a method provided by the scipy.sparse module in the SciPy library. It's used to convert a sparse matrix to the Compressed Sparse Column (CSC) format.
            X.sort_indices()  # This is done to ensure that the indices of non-zero elements within the matrix are sorted in ascending order.

            if X.data.dtype != "float32":
                X.data = np.ascontiguousarray(X.data, dtype=ivy.float32)

            if X.indices.dtype != "int32" or X.indptr.dtype != "int32":
                raise ValueError("No support for np.int64 index-based sparse matrices")

        elif X.dtype != "float32":
            # since we have to copy, we will make it Fortran for efficiency
            X = np.asfortranarray(X, dtype="float32")

        # TODO: This check for y seems to be redundant, as it is also
        #  present in the BaseDecisionTree's fit method, and therefore
        #  can be removed.
        if y.dtype != "float32" or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype="float32")

        if sample_weight is not None and (
            sample_weight.base.dtype != "float64"
            or not sample_weight.base.flags.contiguous
        ):
            sample_weight = ivy.asarray(sample_weight, dtype="float32", order="C")

        return X, y, sample_weight


# Depth first builder ---------------------------------------------------------
# A record on the stack for depth-first tree growing


class StackRecord:
    def __init__(
        self,
        start: int,
        end: int,
        depth: int,
        parent: int,
        is_left: int,
        impurity: float,
        n_constant_features: int,
    ):
        self.start = start
        self.end = end
        self.depth = depth
        self.parent = parent
        self.is_left = is_left
        self.impurity = impurity
        self.n_constant_features = n_constant_features


class FrontierRecord:
    # Record of information of a Node, the frontier for a split. Those records are
    # maintained in a heap to access the Node with the best improvement in impurity,
    # allowing growing trees greedily on this improvement.
    def __init__(
        self,
        node_id: int,
        start: int,
        end: int,
        pos: int,
        depth: int,
        is_leaf: int,
        impurity: float,
        impurity_left: float,
        impurity_right: float,
        improvement: float,
    ):
        self.node_id = node_id
        self.start = start
        self.end = end
        self.pos = pos
        self.depth = depth
        self.is_leaf = is_leaf
        self.impurity = impurity
        self.impurity_left = impurity_left
        self.impurity_right = impurity_right
        self.improvement = improvement


class _CCPPruneController:
    """Base class used by build_pruned_tree_ccp and ccp_pruning_path to control
    pruning."""

    def stop_pruning(
        self,
        effective_alpha: float,
    ):
        """Return 1 to stop pruning and 0 to continue pruning."""
        return 0

    def save_metrics(
        self,
        effective_alpha: float,
        subtree_impurities: float,
    ):
        """Save metrics when pruning."""
        pass

    def after_pruning(
        self,
        in_subtree: str,
    ):
        """Called after pruning."""
        pass


class CostComplexityPruningRecord:
    def __init__(
        self,
        node_idx: int,
        parent: int,
    ):
        self.node_idx = node_idx
        self.parent = parent


class BuildPrunedRecord:
    def __init__(
        self,
        start: int,
        depth: int,
        parent: int,
        is_left: int,
    ):
        self.start = start
        self.depth = depth
        self.parent = parent
        self.is_left = is_left


class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __init__(
        self,
        splitter: Splitter,
        min_samples_split: int,
        min_samples_leaf: int,
        min_weight_leaf: float,
        max_depth: int,
        min_impurity_decrease: float,
    ):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    def build(
        self, tree: Tree, X, y, sample_weight=None, missing_values_in_feature_mask=None
    ):
        """Build a decision tree from the training set (X, y)."""

        # Check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # Initial capacity
        init_capacity: int

        if tree.max_depth <= 10:
            init_capacity = int(2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        splitter = self.splitter
        max_depth = self.max_depth
        min_samples_leaf = self.min_samples_leaf
        min_weight_leaf = self.min_weight_leaf
        min_samples_split = self.min_samples_split
        min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)

        start = 0
        end = 0
        depth = 0
        parent = 0
        is_left = 0
        n_node_samples = splitter.n_samples
        weighted_n_node_samples = 0.0
        # split = SplitRecord()
        split = None
        node_id = 0

        impurity = INFINITY
        n_constant_features = 0
        is_leaf = 0
        first = 1
        max_depth_seen = -1
        rc = 0

        builder_stack: list[StackRecord] = []
        # stack_record = StackRecord()

        # Push root node onto stack
        builder_stack.append(
            StackRecord(
                start=0,
                end=splitter.n_samples,
                depth=0,
                parent=_TREE_UNDEFINED,
                is_left=False,
                impurity=INFINITY,
                n_constant_features=0,
            )
        )
        
        while len(builder_stack) > 0:
            stack_record = builder_stack.pop()

            start = stack_record.start
            end = stack_record.end
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left
            impurity = stack_record.impurity
            n_constant_features = stack_record.n_constant_features

            n_node_samples = end - start
            _, weighted_n_node_samples = splitter.node_reset(
                start, end, weighted_n_node_samples
            )

            is_leaf = (
                depth >= max_depth
                or n_node_samples < min_samples_split
                or n_node_samples < 2 * min_samples_leaf
                or weighted_n_node_samples < 2 * min_weight_leaf
            )

            if first:
                impurity = splitter.node_impurity()
                first = 0

            # impurity == 0 with tolerance due to rounding errors
            is_leaf = is_leaf or impurity <= EPSILON

            if not is_leaf:
                #impurity is passed by value in the original code
                splitter.node_split(impurity, split, n_constant_features)
                # If EPSILON=0 in the below comparison, float precision
                # issues stop splitting, producing trees that are
                # dissimilar to v0.18
                is_leaf = (
                    is_leaf
                    or split.pos >= end
                    or (split.improvement + EPSILON < min_impurity_decrease)
                )

            node_id = tree._add_node(
                parent,
                is_left,
                is_leaf,
                split.feature,
                split.threshold,
                impurity,
                n_node_samples,
                weighted_n_node_samples,
                split.missing_go_to_left,
            )

            if node_id == INTPTR_MAX:
                rc = -1
                break

            # Store value for all nodes, to facilitate tree/model
            # inspection and interpretation
            splitter.node_value(tree.value + node_id * tree.value_stride)

            if not is_leaf:
                # Push right child on stack
                builder_stack.append(
                    StackRecord(
                        start=split.pos,
                        end=end,
                        depth=depth + 1,
                        parent=node_id,
                        is_left=False,
                        impurity=split.impurity_right,
                        n_constant_features=n_constant_features,
                    )
                )

                # Push left child on stack
                builder_stack.append(
                    StackRecord(
                        start=start,
                        end=split.pos,
                        depth=depth + 1,
                        parent=node_id,
                        is_left=True,
                        impurity=split.impurity_left,
                        n_constant_features=n_constant_features,
                    )
                )

            if depth > max_depth_seen:
                max_depth_seen = depth

        if rc >= 0:
            rc = tree._resize_c(tree.node_count)

        if rc == -1:
            raise MemoryError()


class BestFirstTreeBuilder(TreeBuilder):
    """
    Build a decision tree in best-first fashion.

    The best node to expand is given by the node at the frontier that
    has the highest impurity improvement.
    """

    max_leaf_nodes = 0

    def __init__(
        self,
        splitter: Splitter,
        min_samples_split: int,
        min_samples_leaf: int,
        min_weight_leaf: int,
        max_depth: int,
        max_leaf_nodes: int,
        min_impurity_decrease: float,
    ):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    def build(
        self, tree: Tree, X, y, sample_weight=None, missing_values_in_feature_mask=None
    ):
        """Build a decision tree from the training set (X, y)."""

        # Check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # Parameters
        splitter = self.splitter
        max_leaf_nodes = self.max_leaf_nodes

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)

        frontier: list[FrontierRecord] = []
        record = FrontierRecord()
        split_node_left = FrontierRecord()
        split_node_right = FrontierRecord()

        n_node_samples = splitter.n_samples
        max_split_nodes = max_leaf_nodes - 1
        is_leaf = 0
        rc = 0
        node = Node()
        max_depth_seen = -1

        # Initial capacity
        init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize(init_capacity)

        # Add root to frontier
        rc = self._add_split_node(
            splitter,
            tree,
            0,
            n_node_samples,
            INFINITY,
            IS_FIRST,
            IS_LEFT,
            None,
            0,
            split_node_left,
        )

        if rc >= 0:
            _add_to_frontier(split_node_left, frontier)

        while len(frontier) > 0:
            sorted_frontier: list = sorted(frontier, key=_compare_records)

            # The first element in sorted_frontier now has the smallest improvement
            record: FrontierRecord = sorted_frontier[0]

            sorted_frontier.pop(0)
            frontier = sorted_frontier
            del sorted_frontier

            node = tree.nodes[record.node_id]
            is_leaf = record.is_leaf or max_split_nodes <= 0

            if is_leaf:
                # Node is not expandable; set node as leaf
                node.left_child = _TREE_LEAF
                node.right_child = _TREE_LEAF
                node.feature = _TREE_UNDEFINED
                node.threshold = _TREE_UNDEFINED

            else:
                # Node is expandable

                # Decrement number of split nodes available
                max_split_nodes -= 1

                # Compute left split node
                rc = self._add_split_node(
                    splitter,
                    tree,
                    record.start,
                    record.pos,
                    record.impurity_left,
                    IS_NOT_FIRST,
                    IS_LEFT,
                    node,
                    record.depth + 1,
                    split_node_left,
                )

                if rc == -1:
                    break

                node = tree.nodes[record.node_id]

                # Compute right split node
                split_node_right = self._add_split_node(
                    splitter,
                    tree,
                    record.pos,
                    record.end,
                    record.impurity_right,
                    IS_NOT_FIRST,
                    IS_NOT_LEFT,
                    node,
                    record.depth + 1,
                    split_node_right,
                )

                if rc == -1:
                    break

                # Add nodes to queue
                _add_to_frontier(split_node_left, frontier)
                _add_to_frontier(split_node_right, frontier)

            if record.depth > max_depth_seen:
                max_depth_seen = record.depth

        if rc >= 0:
            rc = tree._resize_c(tree.node_count)

        if rc >= 0:
            tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

    def _add_split_node(
        self,
        splitter: Splitter,
        tree: Tree,
        start: int,
        end: int,
        impurity: float,
        is_first: int,
        is_left: int,
        parent: Node,
        depth: int,
        res: FrontierRecord,
    ):
        """Adds node with partition [start, end) to the frontier."""
        split = SplitRecord()
        node_id = None
        n_node_samples = None
        n_constant_features = 0
        min_impurity_decrease = self.min_impurity_decrease
        weighted_n_node_samples = 0.0
        is_leaf = False

        splitter.node_reset(start, end, weighted_n_node_samples)

        if is_first:
            impurity = splitter.node_impurity()

        n_node_samples = end - start
        is_leaf = (
            depth >= self.max_depth
            or n_node_samples < self.min_samples_split
            or n_node_samples < 2 * self.min_samples_leaf
            or weighted_n_node_samples < 2 * self.min_weight_leaf
            or impurity <= EPSILON  # impurity == 0 with tolerance
        )

        if not is_leaf:
            splitter.node_split(impurity, split, n_constant_features)
            is_leaf = (
                is_leaf
                or split.pos >= end
                or split.improvement + EPSILON < min_impurity_decrease
            )

        node_id = tree._add_node(
            parent - tree.nodes if parent is not None else _TREE_UNDEFINED,
            is_left,
            is_leaf,
            split.feature,
            split.threshold,
            impurity,
            n_node_samples,
            weighted_n_node_samples,
            split.missing_go_to_left,
        )

        if node_id == INTPTR_MAX:
            return -1

        splitter.node_value(tree.value + node_id * tree.value_stride)

        res.node_id = node_id
        res.start = start
        res.end = end
        res.depth = depth
        res.impurity = impurity

        if not is_leaf:
            res.pos = split.pos
            res.is_leaf = 0
            res.improvement = split.improvement
            res.impurity_left = split.impurity_left
            res.impurity_right = split.impurity_right
        else:
            res.pos = end
            res.is_leaf = 1
            res.improvement = 0.0
            res.impurity_left = impurity
            res.impurity_right = impurity

        return 0


class _AlphaPruner(_CCPPruneController):
    """Use alpha to control when to stop pruning."""

    def __init__(
        self,
        ccp_alpha: float,
        capacity: int = 0,
    ):
        self.ccp_alpha = ccp_alpha
        self.capacity = capacity

    def stop_pruning(
        self,
        effective_alpha: float,
    ):
        # The subtree on the previous iteration has the greatest ccp_alpha
        # less than or equal to self.ccp_alpha
        return self.ccp_alpha < effective_alpha

    def after_pruning(
        self,
        in_subtree: str,
    ):
        """Updates the number of leaves in subtree."""
        for i in range(in_subtree.shape[0]):
            if in_subtree[i]:
                self.capacity += 1


class _PathFinder(_CCPPruneController):
    """Record metrics used to return the cost complexity path."""

    ccp_alphas = []
    impurities = []
    count = 0

    def __init__(self, node_count: int):
        self.ccp_alphas = ivy.zeros(shape=(node_count), dtype=ivy.float64)
        self.impurities = ivy.zeros(shape=(node_count), dtype=ivy.float64)
        self.count = 0

    def save_metrics(
        self,
        effective_alpha: float,
        subtree_impurities: float,
    ):
        self.ccp_alphas[self.count] = effective_alpha
        self.impurities[self.count] = subtree_impurities
        self.count += 1


# --- Helpers --- #
# --------------- #


def _add_to_frontier(
    rec: FrontierRecord,
    frontier: list,
):
    """Adds record `rec` to the priority queue `frontier`"""
    frontier.append(rec)
    frontier.sort(key=lambda x: x.improvement, reverse=True)


def _build_pruned_tree(
    tree: Tree, orig_tree: Tree, leaves_in_subtree: str, capacity: int
):
    """
    Build a pruned tree.

    Build a pruned tree from the original tree by transforming the nodes in
    ``leaves_in_subtree`` into leaves.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    leaves_in_subtree : unsigned char memoryview, shape=(node_count, )
        Boolean mask for leaves to include in subtree
    capacity : SIZE_t
        Number of nodes to initially allocate in pruned tree
    """
    tree._resize(capacity)

    orig_node_id = 0
    new_node_id = 0
    depth = 0
    parent = 0
    is_left = 0
    is_leaf = 0

    # value_stride for original tree and new tree are the same
    value_stride = orig_tree.value_stride
    max_depth_seen = -1
    rc = 0
    node = Node()
    orig_value_ptr = 0.0
    new_value_ptr = 0.0

    prune_stack: list[BuildPrunedRecord] = []
    stack_record = BuildPrunedRecord()

    prune_stack.append(
        BuildPrunedRecord(
            start=0,
            depth=0,
            parent=_TREE_UNDEFINED,
            is_left=0,
        )
    )

    while len(prune_stack) > 0:
        stack_record = prune_stack.pop()

        orig_node_id = stack_record.start
        depth = stack_record.depth
        parent = stack_record.parent
        is_left = stack_record.is_left

        is_left = leaves_in_subtree[orig_node_id]
        node = orig_tree.nodes[orig_node_id]

        new_node_id = tree._add_node(
            parent,
            is_left,
            is_leaf,
            node.feature,
            node.threshold,
            node.impurity,
            node.n_node_samples,
            node.weighted_n_node_samples,
            node.missing_go_to_left,
        )

        if new_node_id == INTPTR_MAX:
            rc = -1
            break

        # copy value from original tree to new tree
        orig_value_ptr = orig_tree.value + value_stride * orig_node_id
        new_value_ptr = tree.value + value_stride * new_node_id

        new_value_ptr[:value_stride] = orig_value_ptr[:value_stride]

        if not is_leaf:
            # Push right child on stack
            prune_stack.append(
                BuildPrunedRecord(
                    start=node.right_child,
                    depth=depth + 1,
                    parent=new_node_id,
                    is_left=0,
                )
            )

            # push left child on stack
            prune_stack.append(
                BuildPrunedRecord(
                    start=node.left_child,
                    depth=depth + 1,
                    parent=new_node_id,
                    is_left=1,
                )
            )

        if depth > max_depth_seen:
            max_depth_seen = depth

    if rc >= 0:
        tree.max_depth = max_depth_seen

    if rc == -1:
        raise MemoryError("Pruning Tree")


def _build_pruned_tree_ccp(
    tree: Tree,
    orig_tree: Tree,
    ccp_alpha: float,
):
    """
    Build a pruned tree from the original tree using cost complexity pruning.

    The values and nodes from the original tree are copied into the pruned
    tree.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    ccp_alpha : positive double
        Complexity parameter. The subtree with the largest cost complexity
        that is smaller than ``ccp_alpha`` will be chosen. By default,
        no pruning is performed.
    """
    n_nodes = orig_tree.node_count
    leaves_in_subtree = ivy.zeros(shape=n_nodes, dtype=ivy.uint8)

    pruning_controller = _AlphaPruner(ccp_alpha=ccp_alpha)

    _cost_complexity_prune(leaves_in_subtree, orig_tree, pruning_controller)
    _build_pruned_tree(tree, orig_tree, leaves_in_subtree, pruning_controller.capacity)


def _check_n_classes(n_classes, expected_dtype):
    if n_classes.ndim != 1:
        raise ValueError(
            "Wrong dimensions for n_classes from the pickle: "
            f"expected 1, got {n_classes.ndim}"
        )

    if n_classes.dtype == expected_dtype:
        return n_classes

    # Handles both different endianness and different bitness
    if n_classes.dtype.kind == "i" and n_classes.dtype.itemsize in [4, 8]:
        return n_classes.astype(expected_dtype, casting="same_kind")

    raise ValueError(
        "n_classes from the pickle has an incompatible dtype:\n"
        f"- expected: {expected_dtype}\n"
        f"- got:      {n_classes.dtype}"
    )


def _compare_records(
    left: FrontierRecord,
    right: FrontierRecord,
):
    return left.improvement < right.improvement


def _cost_complexity_prune(
    leaves_in_subtree: str,
    orig_tree: Tree,
    controller: _CCPPruneController,
):
    """
    Perform cost complexity pruning.

    This function takes an already grown tree, `orig_tree` and outputs a
    boolean mask `leaves_in_subtree` which are the leaves in the pruned tree.
    During the pruning process, the controller is passed the effective alpha and
    the subtree impurities. Furthermore, the controller signals when to stop
    pruning.

    Parameters
    ----------
    leaves_in_subtree : unsigned char[:]
        Output for leaves of subtree
    orig_tree : Tree
        Original tree
    ccp_controller : _CCPPruneController
        Cost complexity controller
    """
    i = 0
    n_nodes = orig_tree.node_count
    # prior probability using weighted samples
    weighted_n_nodes_samples = orig_tree.weighted_n_node_samples
    total_sum_weights = weighted_n_nodes_samples[0]
    impurity = orig_tree.impurity
    # weighted impurity of each node
    r_node = ivy.empty(shape=n_nodes, dtype=ivy.float64)

    child_l = orig_tree.children_left
    child_r = orig_tree.children_right
    parent = ivy.zeros(shape=n_nodes, dtype=ivy.int16)

    ccp_stack: list[CostComplexityPruningRecord] = []
    stack_record = CostComplexityPruningRecord()
    node_idx = 0
    node_indices_stack: list[int] = []

    n_leaves = ivy.zeros(shape=n_nodes, dtype=ivy.int16)
    r_branch = ivy.zeros(shape=n_nodes, dtype=ivy.float64)
    current_r = 0.0
    leaf_idx = 0
    parent_idx = 0

    # candidate nodes that can be pruned
    candidate_nodes = ivy.zeros(shape=n_nodes, dtype=ivy.uint8)

    # nodes in subtree
    in_subtree = ivy.ones(shape=n_nodes, dtype=ivy.uint8)
    pruned_branch_node_idx = 0
    subtree_alpha = 0.0
    effective_alpha = 0.0
    n_pruned_leaves = 0
    r_diff = 0.0
    max_float64 = ivy.finfo(ivy.float64).max

    # find parent node ids and leaves

    for i in range(r_node.shape[0]):
        r_node[i] = weighted_n_nodes_samples[i] * impurity[i] / total_sum_weights

    # Push the root node
    ccp_stack.append(CostComplexityPruningRecord(node_idx=0, parent=_TREE_UNDEFINED))

    while len(ccp_stack) > 0:
        stack_record = ccp_stack.pop()

        node_idx = stack_record.node_idx
        parent_idx = stack_record.parent

        if child_l[node_idx] == _TREE_LEAF:
            # ... and child_r[node_idx] == _TREE_LEAF:
            leaves_in_subtree[node_idx] = 1
        else:
            ccp_stack.append(
                CostComplexityPruningRecord(
                    node_idx=child_l[node_idx],
                    parent=node_idx,
                )
            )
            ccp_stack.append(
                CostComplexityPruningRecord(
                    node_idx=child_r[node_idx],
                    parent=node_idx,
                )
            )

    # computes number of leaves in all branches and the overall impurity of
    # the branch. The overall impurity is the sum of r_node in its leaves.
    for leaf_idx in range(leaves_in_subtree.shape[0]):
        if not leaves_in_subtree[leaf_idx]:
            continue
        r_branch[leaf_idx] = r_node[leaf_idx]

        # bubble up values to ancestor nodes
        current_r = r_node[leaf_idx]
        while leaf_idx != 0:
            parent_idx = parent[leaf_idx]
            r_branch[parent_idx] += current_r
            n_leaves[parent_idx] += 1
            leaf_idx = parent_idx

    for i in range(leaves_in_subtree.shape[0]):
        candidate_nodes[i] = not leaves_in_subtree[i]

    # save metrics before pruning
    controller.save_metrics(0.0, r_branch[0])

    # while root node is not a leaf
    while candidate_nodes[0]:
        # computes ccp_alpha for subtrees and finds the minimal alpha
        effective_alpha = max_float64
        for i in range(n_nodes):
            if not candidate_nodes[i]:
                continue
            subtree_alpha = (r_node[i] - r_branch[i]) / (n_leaves[i] - 1)
            if subtree_alpha < effective_alpha:
                effective_alpha = subtree_alpha
                pruned_branch_node_idx = i

        if controller.stop_pruning(effective_alpha):
            break

        node_indices_stack.append(pruned_branch_node_idx)

        # descendants of branch and not in subtree
        while len(node_indices_stack) > 0:
            node_idx = node_indices_stack.pop()

            if not in_subtree[node_idx]:
                continue  # branch has already been marked for pruning

            candidate_nodes[node_idx] = 0
            leaves_in_subtree[node_idx] = 0
            in_subtree[node_idx] = 0

            if child_l[node_idx] != _TREE_LEAF:
                # branch has already been marked for pruning
                node_indices_stack.append(child_l[node_idx])
                node_indices_stack.append(child_r[node_idx])

        leaves_in_subtree[pruned_branch_node_idx] = 1
        in_subtree[pruned_branch_node_idx] = 1

        # update number of leaves
        n_pruned_leaves = n_leaves[pruned_branch_node_idx] - 1
        n_leaves[pruned_branch_node_idx] = 0

        # computes the increase in r_branch to bubble up
        r_diff = r_node[pruned_branch_node_idx] - r_branch[pruned_branch_node_idx]
        r_branch[pruned_branch_node_idx] = r_node[pruned_branch_node_idx]

        # bubble up values to ancestors
        node_idx = parent[pruned_branch_node_idx]
        while node_idx != _TREE_UNDEFINED:
            n_leaves[node_idx] -= n_pruned_leaves
            r_branch[node_idx] += r_diff
            node_idx = parent[node_idx]

        controller.save_metrics(effective_alpha, r_branch[0])

    controller.after_pruning(in_subtree)


# --- Main --- #
# ------------ #


def ccp_pruning_path(
    orig_tree: Tree,
):
    """
    Computes the cost complexity pruning path.

    Parameters
    ----------
    tree : Tree
        Original tree.

    Returns
    -------
    path_info : dict
        Information about pruning path with attributes:

        ccp_alphas : ndarray
            Effective alphas of subtree during pruning.

        impurities : ndarray
            Sum of the impurities of the subtree leaves for the
            corresponding alpha value in ``ccp_alphas``.
    """
    leaves_in_subtree = ivy.zeros(shape=orig_tree.node_count, dtype=ivy.uint8)

    path_finder = _PathFinder(orig_tree.node_count)

    _cost_complexity_prune(leaves_in_subtree, orig_tree, path_finder)

    total_items = path_finder.count
    ccp_alphas = ivy.empty(shape=total_items, dtype=ivy.float64)
    impurities = ivy.empty(shape=total_items, dtype=ivy.float64)
    count = 0

    while count < total_items:
        ccp_alphas[count] = path_finder.ccp_alphas[count]
        impurities[count] = path_finder.impurities[count]
        count += 1

    return {
        "ccp_alphas": ivy.asarray(ccp_alphas),
        "impurities": ivy.asarray(impurities),
    }


# Define constants
INTPTR_MAX = ivy.iinfo(ivy.int32).max
TREE_UNDEFINED = -2
_TREE_UNDEFINED = TREE_UNDEFINED
TREE_LEAF = -1
_TREE_LEAF = TREE_LEAF
