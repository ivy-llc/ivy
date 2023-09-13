import numpy as np
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse import isspmatrix_csr


# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

# Define constants
INFINITY = np.inf
EPSILON = np.finfo(np.double).eps

# Some handy constants (BestFirstTreeBuilder)
IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
_TREE_LEAF = TREE_LEAF
_TREE_UNDEFINED = TREE_UNDEFINED


# Since you're dealing with Cython-specific types and features,
# it's important to provide a dummy definition for Node.
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


dummy = Node()
# Create a numpy dtype for Node using the dummy object
NODE_DTYPE = np.asarray([dummy], dtype=object).dtype

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
        tree,
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
        """Check input dtype, layout, and format"""
        if issparse(X):
            X = X.tocsc() #tocsc() is a method provided by the scipy.sparse module in the SciPy library. It's used to convert a sparse matrix to the Compressed Sparse Column (CSC) format. 
            X.sort_indices() #This is done to ensure that the indices of non-zero elements within the matrix are sorted in ascending order.

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index-based sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy, we will make it Fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.base.dtype != DTYPE or not y.base.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DTYPE)

        if sample_weight is not None and (
            sample_weight.base.dtype != DOUBLE
            or not sample_weight.base.flags.contiguous
        ):
            sample_weight = np.asarray(sample_weight, dtype=DOUBLE, order="C")

        return X, y, sample_weight


# Depth first builder ---------------------------------------------------------
# A record on the stack for depth-first tree growing
class StackRecord:
    def __init__(self, start, end, depth, parent, is_left, impurity, n_constant_features):
        self.start = start
        self.end = end
        self.depth = depth
        self.parent = parent
        self.is_left = is_left
        self.impurity = impurity
        self.n_constant_features = n_constant_features


class SplitRecord:
    def __init__(
        self,
        feature,
        pos,
        threshold,
        improvement,
        impurity_left,
        impurity_right,
        missing_go_to_left,
        n_missing,
    ):
        self.feature = feature
        self.pos = pos
        self.threshold = threshold
        self.improvement = improvement
        self.impurity_left = impurity_left
        self.impurity_right = impurity_right
        self.missing_go_to_left = missing_go_to_left
        self.n_missing = n_missing


class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __init__(
        self, splitter, min_samples_split,
        min_samples_leaf, min_weight_leaf,
        max_depth, min_impurity_decrease
    ):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    def build(
        self,
        tree,
        X,
        y,
        sample_weight=None,
        missing_values_in_feature_mask=None
    ):
        """Build a decision tree from the training set (X, y)."""

        # Check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # Initial capacity
        init_capacity = (2 ** (tree.max_depth + 1)) - 1 if tree.max_depth <= 10 else 2047

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

        stack = []

        # Push root node onto stack
        stack.append(
            StackRecord(
                start=0,
                end=splitter.n_samples,
                depth=0,
                parent=_TREE_UNDEFINED,
                is_left=False,
                impurity=INFINITY,
                n_constant_features=0
            )
        )
        weighted_n_node_samples = np.zeros(1, dtype=np.double)
        while stack:
            stack_record = stack.pop()

            start = stack_record.start
            end = stack_record.end
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left
            impurity = stack_record.impurity
            n_constant_features = stack_record.n_constant_features

            n_node_samples = end - start
            splitter.node_reset(start, end, weighted_n_node_samples)

            is_leaf = (
                depth >= max_depth
                or n_node_samples < min_samples_split
                or n_node_samples < 2 * min_samples_leaf
                or np.sum(sample_weight[start:end]) < 2 * min_weight_leaf
            )

            if is_left:
                impurity = splitter.node_impurity()

            is_leaf = is_leaf or impurity <= EPSILON

            if not is_leaf:
                split = SplitRecord()  # No idea what is SplitRecord in original code. Maybe this never gets called, not sure
                splitter.node_split(impurity, split, n_constant_features)
                is_leaf = (
                    is_leaf
                    or split.pos >= end
                    or (split.improvement + EPSILON < min_impurity_decrease)
                )

            node_id = tree._add_node(
                parent,
                is_left,
                is_leaf,
                split.feature if not is_leaf else 0,
                split.threshold if not is_leaf else 0,
                impurity,
                n_node_samples,
                np.sum(sample_weight[start:end]),
                split.missing_go_to_left,
            )

            if node_id == np.iinfo(np.intp).max:
                raise MemoryError()

            splitter.node_value(tree.value + node_id * tree.value_stride)

            if not is_leaf:
                # Push right child on stack
                stack.append(
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
                stack.append(
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


class Tree:
    def __init__(self, n_features, n_classes, n_outputs):
        """Constructor."""
        self.n_features = None
        self.n_outputs = None
        self.n_classes = None
        self.max_n_classes = None
        self.max_depth = None
        self.node_count = None
        self.capacity = None
        self.nodes = None
        self.value = None
        self.value_stride = None

        dummy = 0
        size_t_dtype = np.array(dummy).dtype

        n_classes = _check_n_classes(n_classes, size_t_dtype)

        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = np.zeros(n_outputs, dtype=size_t_dtype)

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

    # NOT CONSIDERING PICKINLING FOR NOW
    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        raise NotImplementedError

    # NOT CONSIDERING PICKINLING FOR NOW
    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    # NOT CONSIDERING PICKINLING FOR NOW
    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        raise NotImplementedError

    def _resize(self, capacity):
        """
        Resize all inner arrays to `capacity`. If `capacity` is -1, then double the size of the inner arrays.
        Returns -1 in case of failure to allocate memory (and raise MemoryError), or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Raise MemoryError if resizing fails
            raise MemoryError()

    def _resize_c(self, capacity=float('inf')):
        """
        Guts of _resize
        Returns -1 in case of failure to allocate memory (and raise MemoryError),
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes is not None:
            return 0

        if capacity == float('inf'):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        # This section is relevant if the code is dealing with C arrays.
        # In Python, resizing arrays is handled automatically by lists or numpy arrays.
        # You won't need to explicitly reallocate memory or initialize values like this.
        # replaced safe_realloc(&self.nodes, capacity) with the following
        new_nodes = np.empty(capacity, dtype=object)
        for i in range(len(self.nodes)):
            new_nodes[i] = self.nodes[i]
        self.nodes = new_nodes

        # replaced safe_realloc(&self.value, capacity * self.value_stride) with the following
        new_value = np.empty(capacity * self.value_stride, dtype=object)
        for i in range(len(self.value)):
            new_value[i] = self.value[i]
        self.value = new_value

        # if capacity smaller than node_count, adjust the counter
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

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return -1

        node = self.nodes[node_id]
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
        """Predict target for X."""
        out = self._get_value_ndarray()[self.apply(X), :, :]

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
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s" % type(X))

        if X.dtype != np.float32:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        n_samples, n_features = X.shape

        # Initialize output
        out = np.zeros(n_samples, dtype=np.intp)

        with np.nditer(X, flags=['c_index', 'multi_index'], op_flags=['readonly'], order='C') as it:
            for x_value, index in it:
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    X_i_node_feature = x_value
                    # ... and node.right_child != _TREE_LEAF:
                    if np.isnan(X_i_node_feature):
                        if node.missing_go_to_left:
                            node = self.nodes[node.left_child]
                        else:
                            node = self.nodes[node.right_child]
                    elif X_i_node_feature <= node.threshold:
                        node = self.nodes[node.left_child]
                    else:
                        node = self.nodes[node.right_child]

                out[index[0]] = node - self.nodes  # node offset

        return out

    def _apply_sparse_csr(self, X):
        """Finds the terminal region (=leaf node) for each sample in sparse X."""
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s" % type(X))

        if X.dtype != np.float32:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        n_samples, n_features = X.shape

        # Initialize output
        out = np.zeros(n_samples, dtype=np.intp)

        # Initialize auxiliary data structures
        feature_to_sample = np.full(n_features, -1, dtype=np.intp)
        X_sample = np.zeros(n_features, dtype=np.float32)

        for i in range(n_samples):
            node = self.nodes

            for k in range(X.indptr[i], X.indptr[i + 1]):
                feature_to_sample[X.indices[k]] = i
                X_sample[X.indices[k]] = X.data[k]

            while node.left_child != _TREE_LEAF:
                if feature_to_sample[node.feature] == i:
                    feature_value = X_sample[node.feature]
                else:
                    feature_value = 0.0

                if feature_value <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]

            out[i] = node - self.nodes  # node offset

        return out

    def decision_path(self, X):
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    def _decision_path_dense(self, X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s" % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        X_ndarray = X
        n_samples = X.shape[0]

        # Initialize output
        indptr = np.zeros(n_samples + 1, dtype=np.intp)
        indices = np.zeros(n_samples * (1 + self.max_depth), dtype=np.intp)

        # Initialize auxiliary data-structure
        node = None
        i = 0

        for i in range(n_samples):
            node = self.nodes
            indptr[i + 1] = indptr[i]

            # Add all external nodes
            while node.left_child != _TREE_LEAF:
                # ... and node.right_child != _TREE_LEAF:
                indices[indptr[i + 1]] = node - self.nodes
                indptr[i + 1] += 1

                if X_ndarray[i, node.feature] <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]

            # Add the leaf node
            indices[indptr[i + 1]] = node - self.nodes
            indptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        data = np.ones(shape=len(indices), dtype=np.intp)
        out = csr_matrix((data, indices, indptr), shape=(n_samples, self.node_count))

        return out

    def _decision_path_sparse_csr(self, X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isspmatrix_csr(X):
            raise ValueError("X should be in csr_matrix format, got %s" % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Initialize output
        indptr = np.zeros(n_samples + 1, dtype=np.intp)
        indices = np.zeros(n_samples * (1 + self.max_depth), dtype=np.intp)

        # Initialize auxiliary data-structure
        feature_value = 0.0
        node = None
        X_sample = np.zeros(n_features, dtype=DTYPE)
        feature_to_sample = np.full(n_features, -1, dtype=np.intp)

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

        indices = indices[:indptr[n_samples]]
        data = np.ones(shape=len(indices), dtype=np.intp)
        out = csr_matrix((data, indices, indptr), shape=(n_samples, self.node_count))

        return out

    def compute_node_depths(self):
        """
        Compute the depth of each node in a tree.

        .. versionadded:: 1.3

        Returns
        -------
        depths : ndarray of shape (self.node_count,), dtype=np.int64
            The depth of each node in the tree.
        """
        depths = np.empty(self.node_count, dtype=np.int64)
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

    def compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        nodes = self.nodes
        node = nodes
        end_node = node + self.node_count

        importances = np.zeros(self.n_features, dtype=np.float64)

        while node != end_node:
            if node.left_child != _TREE_LEAF:
                left = nodes[node.left_child]
                right = nodes[node.right_child]

                importances[node.feature] += (
                    node.weighted_n_node_samples * node.impurity
                    - left.weighted_n_node_samples * left.impurity
                    - right.weighted_n_node_samples * right.impurity
                )
            node += 1

        for i in range(self.n_features):
            importances[i] /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    def _get_value_ndarray(self):
        """
        Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the
        underlying memory.
        """
        shape = (self.node_count, self.n_outputs, self.max_n_classes)
        arr = np.ndarray(shape, dtype=np.float64, buffer=self.value)
        arr.base = self
        return arr

    def _get_node_ndarray(self):
        """
        Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the
        underlying memory. Individual fields are publicly accessible as
        properties of the Tree.
        """
        shape = (self.node_count,)
        dtype = np.dtype(
            [
                ("left_child", np.intp),
                ("right_child", np.intp),
                ("feature", np.intp),
                ("threshold", np.float64),
                ("impurity", np.float64),
                ("n_node_samples", np.intp),
                ("weighted_n_node_samples", np.float64),
                ("missing_go_to_left", np.uint8),
            ]
        )
        arr = np.ndarray(shape, dtype=dtype, buffer=self.nodes)
        arr.base = self
        return arr

    def compute_partial_dependence(self, X, target_features, out):
        out.fill(0.0)  # Initialize the output array

        _TREE_LEAF = self._TREE_LEAF  # The value for leaf nodes

        for sample_idx in range(X.shape[0]):
            stack_size = 1
            node_idx_stack = [0]  # root node
            weight_stack = [1.0]  # all samples are in the root node
            total_weight = 0.0

            while stack_size > 0:
                stack_size -= 1
                current_node_idx = node_idx_stack[stack_size]
                current_node = self.nodes[current_node_idx]

                if current_node.left_child == _TREE_LEAF:
                    # Leaf node
                    out[sample_idx] += weight_stack[stack_size] * self.value[current_node_idx]
                    total_weight += weight_stack[stack_size]
                else:
                    is_target_feature = any(target_feature == current_node.feature for target_feature in target_features)
                    if is_target_feature:
                        if X[sample_idx, current_node.feature] <= current_node.threshold:
                            node_idx_stack.append(current_node.left_child)
                            weight_stack.append(weight_stack[stack_size])
                            stack_size += 1
                        else:
                            node_idx_stack.append(current_node.right_child)
                            weight_stack.append(weight_stack[stack_size])
                            stack_size += 1
                    else:
                        left_sample_frac = self.nodes[current_node.left_child].weighted_n_node_samples / current_node.weighted_n_node_samples
                        current_weight = weight_stack[stack_size]
                        node_idx_stack.extend([current_node.left_child, current_node.right_child])
                        weight_stack.extend([current_weight * left_sample_frac, current_weight * (1 - left_sample_frac)])
                        stack_size += 2

            if not (0.999 < total_weight < 1.001):
                raise ValueError(f"Total weight should be 1.0 but was {total_weight:.9f}")


class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __init__(
        self,
        splitter,
        min_samples_split,
        min_samples_leaf,
        min_weight_leaf,
        max_depth,
        min_impurity_decrease,
    ):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    def build(
        self, tree, X, y, sample_weight=None, missing_values_in_feature_mask=None
    ):
        """Build a decision tree from the training set (X, y)."""

        # Check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # Initial capacity
        init_capacity = (
            (2 ** (tree.max_depth + 1)) - 1 if tree.max_depth <= 10 else 2047
        )

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

        stack = []

        # Push root node onto stack
        stack.append(
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
        weighted_n_node_samples = np.zeros(1, dtype=np.double)
        while stack:
            stack_record = stack.pop()

            start = stack_record.start
            end = stack_record.end
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left
            impurity = stack_record.impurity
            n_constant_features = stack_record.n_constant_features

            n_node_samples = end - start
            splitter.node_reset(start, end, weighted_n_node_samples)

            is_leaf = (
                depth >= max_depth
                or n_node_samples < min_samples_split
                or n_node_samples < 2 * min_samples_leaf
                or np.sum(sample_weight[start:end]) < 2 * min_weight_leaf
            )

            if is_left:
                impurity = splitter.node_impurity()

            is_leaf = is_leaf or impurity <= EPSILON

            if not is_leaf:
                split = (
                    SplitRecord()
                )  # No idea what is SplitRecord in original code. Maybe this never gets called, not sure
                splitter.node_split(impurity, split, n_constant_features)
                is_leaf = (
                    is_leaf
                    or split.pos >= end
                    or (split.improvement + EPSILON < min_impurity_decrease)
                )

            node_id = tree._add_node(
                parent,
                is_left,
                is_leaf,
                split.feature if not is_leaf else 0,
                split.threshold if not is_leaf else 0,
                impurity,
                n_node_samples,
                np.sum(sample_weight[start:end]),
                split.missing_go_to_left,
            )

            if node_id == np.iinfo(np.intp).max:
                raise MemoryError()

            splitter.node_value(tree.value + node_id * tree.value_stride)

            if not is_leaf:
                # Push right child on stack
                stack.append(
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
                stack.append(
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


# --- Helpers --- #
# --------------- #


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


dummy = Node()
# Create a numpy dtype for Node using the dummy object
NODE_DTYPE = np.asarray([dummy], dtype=object).dtype
_TREE_LEAF = TREE_LEAF
_TREE_UNDEFINED = TREE_UNDEFINED
