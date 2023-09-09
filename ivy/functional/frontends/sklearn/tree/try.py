import ivy 

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

    def _resize_c(self, capacity=float('inf')):
        raise NotImplementedError

    def _add_node(self, parent, is_left, is_leaf, feature, threshold, impurity, n_node_samples, weighted_n_node_samples, missing_go_to_left):
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
        out = internal_data[predictions] #not sure if this accurately translates to .take(self.apply(X), axis=0, mode='clip')
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
                    feature_value = ivy.array( 0, dtype="float32") #feature value is computed during training

                threshold = ivy.array(node.threshold, dtype="float32")
                if feature_value <= threshold:
                    node = node.left_child
                else:
                    node = node.right_child

            # Get the index of the leaf node
            out[i] = self.nodes.index(node)

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
        """Compute the depth of each node in a tree.

        .. versionadded:: 1.3

        Returns
        -------
        depths : ivy of shape (self.node_count,), dtype="int32"
            The depth of each node in the tree.
        """
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

    
    '''
    This code is typically used after fitting a decision tree model to assess the importance of each feature in making predictions.
    The feature importances indicate the contribution of each feature to the overall decision-making process of the tree.
    '''
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
                    node.weighted_n_node_samples * node.impurity -
                    left.weighted_n_node_samples * left.impurity -
                    right.weighted_n_node_samples * right.impurity
                )

                importances[node.feature] += importance

        if normalize:
            total_importance = ivy.sum(importances)

            if total_importance > 0.0:
                # Normalize feature importances to sum up to 1.0
                importances /= total_importance

        return importances
    
    def _get_value_ndarray(self):
        """Wraps value as a 3-dimensional NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        shape = (
            int(self.node_count),
            int(self.n_outputs),
            int(self.max_n_classes)
        )
        arr = ivy.array(self.value, dtype="float32").reshape(shape)
        return arr
    
    '''
    this code creates a ivy structured array that wraps the internal nodes of a decision tree. 
    This array can be used to access and manipulate the tree's nodes efficiently in Python.
    '''
    def _get_node_tensor(self):
        """Wraps nodes as a PyTorch tensor.

        The tensor keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the Tree.
        """
        
        # Create a tensor with a custom data type for the tree nodes
        nodes_tensor = ivy.zeros(self.node_count, dtype="float32")

        # Fill the tensor with node data
        for i, node in enumerate(self.nodes):
            nodes_tensor[i] = ivy.array((
                node.impurity,
                node.n_node_samples,
                node.weighted_n_node_samples,
                node.left_child,
                node.right_child,
                node.feature,
                node.threshold,
                node.missing_go_to_left,
            ), dtype="float32")

        # Attach a reference to the tree
        nodes_tensor.tree_reference = self

        return nodes_tensor
    
    '''
    This code effectively computes the partial dependence values for a set of grid points based on a given decision tree. 
    Partial dependence helps understand how the model's predictions change with variations in specific features while keeping other features constant.
    '''   
    def compute_partial_dependence(self, X, target_features, out):
        """
        Partial dependence of the response on the target_feature set.

        For each sample in X, a tree traversal is performed.
        Each traversal starts from the root with weight 1.0.

        At each non-leaf node that splits on a target feature, either the left child or the right child is visited based on the feature value of the current sample, and the weight is not modified.
        At each non-leaf node that splits on a complementary feature, both children are visited, and the weight is multiplied by the fraction of training samples that went to each child.

        At each leaf, the value of the node is multiplied by the current weight (weights sum to 1 for all visited terminal nodes).

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be evaluated.
        target_features : numpy.ndarray, shape (n_target_features)
            The set of target features for which the partial dependence should be evaluated.
        out : numpy.ndarray, shape (n_samples)
            The value of the partial dependence function on each grid point.
        """
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
        if not (0.999 < np.sum(weight_stack) < 1.001):
            raise ValueError("Total weight should be 1.0 but was %.9f" % np.sum(weight_stack))


def _check_n_classes(n_classes, expected_dtype):
    if n_classes.ndim != 1:
        raise ValueError(
            f"Wrong dimensions for n_classes from the pickle: "
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






