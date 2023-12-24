import ivy
from ._splitter import SplitRecord

EPSILON = ivy.finfo(ivy.double).eps
INFINITY = ivy.inf
INTPTR_MAX = ivy.iinfo(ivy.int32).max
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


class Tree:
    def __init__(self, n_features, n_classes, n_outputs):
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.nodes = []
        self.value = None

        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = ivy.zeros(n_outputs, dtype=ivy.int32)

        self.max_n_classes = ivy.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

    def _resize(self, capacity):
        self._resize_c(capacity)

    def _resize_c(self, capacity=INTPTR_MAX):
        if capacity == self.capacity and len(self.nodes) != 0:
            return 0
        if capacity == INTPTR_MAX:
            if self.capacity == 0:
                capacity = 3
            else:
                capacity = 2 * self.capacity
        if self.value is None:
            self.value = ivy.zeros(
                (capacity, int(self.n_outputs), int(self.max_n_classes)),
                dtype=ivy.float32,
            )
        else:
            self.value = ivy.concat(
                [
                    self.value,
                    ivy.zeros(
                        (
                            int(capacity - self.capacity),
                            int(self.n_outputs),
                            int(self.max_n_classes),
                        ),
                        dtype=ivy.float32,
                    ),
                ]
            )
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
        node_id = self.node_count
        if node_id >= self.capacity:
            self._resize_c()

        node = Node()
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
            node.feature = feature
            node.threshold = threshold
            node.missing_go_to_left = missing_go_to_left

        self.nodes.append(node)
        self.node_count += 1

        return node_id

    def predict(self, X):
        X_applied = self.apply(X)
        out = ivy.take(self.value, X_applied, axis=0)
        if self.n_outputs == 1:
            out = out.reshape((X.shape[0], self.max_n_classes))
        return out

    def apply(self, X):
        return self._apply_dense(X)

    def _apply_dense(self, X):
        X_tensor = X
        n_samples = X.shape[0]
        out = ivy.zeros(n_samples, dtype="int32")
        for i in range(n_samples):
            node = self.nodes[0]  # root node
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
            out[i] = self.nodes.index(node)  # terminal node index
        return out


class StackRecord:
    def __init__(
        self,
        start,
        end,
        depth,
        parent,
        is_left,
        impurity,
        n_constant_features,
    ):
        self.start = start
        self.end = end
        self.depth = depth
        self.parent = parent
        self.is_left = is_left
        self.impurity = impurity
        self.n_constant_features = n_constant_features


class TreeBuilder:
    def build(
        self,
        tree,
        X,
        y,
        sample_weight=None,
        missing_values_in_feature_mask=None,
    ):
        pass


class DepthFirstTreeBuilder(TreeBuilder):
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
        if tree.max_depth <= 10:
            init_capacity = int(2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047
        tree._resize(init_capacity)

        splitter = self.splitter
        max_depth = self.max_depth
        min_samples_leaf = self.min_samples_leaf
        min_weight_leaf = self.min_weight_leaf
        min_samples_split = self.min_samples_split
        min_impurity_decrease = self.min_impurity_decrease

        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)
        weighted_n_node_samples = 0.0
        split = SplitRecord()
        first = 1
        max_depth_seen = -1
        builder_stack = []

        # Push root node onto stack
        builder_stack.append(
            StackRecord(
                start=0,
                end=splitter.n_samples,
                depth=0,
                parent=-2,
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
            is_leaf = is_leaf or impurity <= EPSILON

            if not is_leaf:
                _, n_constant_features, split = splitter.node_split(
                    impurity, split, n_constant_features
                )

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
            tree.value = splitter.node_value(tree.value, node_id)

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
                tree.max_depth = max_depth_seen
