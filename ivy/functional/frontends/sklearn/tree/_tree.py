import ivy
from ._splitter import SplitRecord

EPSILON = ivy.finfo(ivy.double).eps
INFINITY = ivy.inf


class StackRecord: ...


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
