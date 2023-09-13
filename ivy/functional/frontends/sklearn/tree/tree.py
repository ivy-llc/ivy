import ivy

from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse import isspmatrix_csr

from ._splitter import SplitRecord

# Define constants
INFINITY = ivy.inf
EPSILON = ivy.finfo(ivy.double).eps

# Some handy constants (BestFirstTreeBuilder)
IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
_TREE_LEAF = TREE_LEAF
_TREE_UNDEFINED = TREE_UNDEFINED


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

            if X.data.dtype != "float32":
                X.data = ivy.ascontiguousarray(X.data, dtype="float32")

            if X.indices.dtype != "int32" or X.indptr.dtype != "int32":
                raise ValueError("No support for np.int64 index-based sparse matrices")

        elif X.dtype != "float32":
            # since we have to copy, we will make it Fortran for efficiency
            X = ivy.asfortranarray(X, dtype="float32")

        if y.base.dtype != "float32" or not y.base.flags.contiguous:
            y = ivy.ascontiguousarray(y, dtype="float32")

        if sample_weight is not None and (
            sample_weight.base.dtype != "float32"
            or not sample_weight.base.flags.contiguous
        ):
            sample_weight = ivy.asarray(sample_weight, dtype="float32", order="C")

        return X, y, sample_weight


# Depth first builder ---------------------------------------------------------
# A record on the stack for depth-first tree growing


class StackRecord:
    def __init__(
        self, start, end, depth, parent, is_left, impurity, n_constant_features
    ):
        self.start = start
        self.end = end
        self.depth = depth
        self.parent = parent
        self.is_left = is_left
        self.impurity = impurity
        self.n_constant_features = n_constant_features


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
        weighted_n_node_samples = ivy.zeros(1, dtype="float32")
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
                or ivy.sum(sample_weight[start:end]) < 2 * min_weight_leaf
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
                ivy.sum(sample_weight[start:end]),
                split.missing_go_to_left,
            )

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
        self.n_classes = None
        self.n_outputs = None
        self.max_n_classes = None
        self.max_depth = None
        self.node_count = None
        self.capacity = None
        self.nodes = []  # replaced it with array since this array will contain nodes
        self.value = None
        self.value_stride = None

        size_t_dtype = "float32"

        n_classes = _check_n_classes(n_classes, size_t_dtype)

        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = ivy.zeros(n_outputs, dtype=size_t_dtype)

        self.max_n_classes = ivy.max(n_classes)
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
        raise NotImplementedError

    def _resize_c(self, capacity=float("inf")):
        raise NotImplementedError

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

        node = Node()  #self.nodes contains a list of nodes, it returns the node at node_id location

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
        out = internal_data[predictions] # not sure if this accurately translates to .take(self.apply(X), axis=0, mode='clip')
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
            raise ValueError("X should be a ivy.data_classes.array.array.Array, got %s" % type(X))

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
    
    #not so sure about sparse implimentation yet
    def _apply_sparse_csr(self, X):
        """Finds the terminal region (=leaf node) for each sample in sparse X."""
        if not isinstance(X, ivy.data_classes.array.array.Array):
            raise ValueError("X should be a ivy.data_classes.array.array.Array, got %s" % type(X))

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
                    feature_value = ivy.array(0, dtype="float32")  # feature value is computed during training

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
            raise ValueError("X should be a ivy.data_classes.array.array.Array, got %s" % type(X))

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

        indices = indices[:indptr[n_samples]]
        data = ivy.ones(indices.shape, dtype="float32")
        #csr_matrix is not implemented
        out = csr_matrix((data, indices, indptr), shape=(n_samples, self.node_count))

        return out
    
    #not tested
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

            
        indices = indices[:indptr[n_samples]]
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
                    out[sample_idx] += weight_stack[stack_size] * self.value[current_node_idx]
                else:  # Non-leaf node
                    is_target_feature = current_node.feature in target_features
                    if is_target_feature:
                        # Push left or right child on the stack
                        if X[sample_idx, current_node.feature] <= current_node.threshold:
                            node_idx_stack[stack_size] = current_node.left_child
                        else:
                            node_idx_stack[stack_size] = current_node.right_child
                        stack_size += 1
                    else:
                        # Push both children onto the stack and give a weight proportional to the number of samples going through each branch
                        left_child = current_node.left_child
                        right_child = current_node.right_child
                        left_sample_frac = self.nodes[left_child].weighted_n_node_samples / current_node.weighted_n_node_samples
                        current_weight = weight_stack[stack_size]
                        weight_stack[stack_size] = current_weight * left_sample_frac
                        stack_size += 1

                        node_idx_stack[stack_size] = right_child
                        weight_stack[stack_size] = current_weight * (1 - left_sample_frac)
                        stack_size += 1

        # Sanity check. Should never happen.
        if not (0.999 < ivy.sum(weight_stack) < 1.001):
            raise ValueError("Total weight should be 1.0 but was %.9f" % ivy.sum(weight_stack))


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
        weighted_n_node_samples = ivy.zeros(1, dtype="float32")
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
                or ivy.sum(sample_weight[start:end]) < 2 * min_weight_leaf
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
                ivy.sum(sample_weight[start:end]),
                split.missing_go_to_left,
            )

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


_TREE_LEAF = TREE_LEAF
_TREE_UNDEFINED = TREE_UNDEFINED