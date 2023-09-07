#splitter.init
#splitter.n_samples
#splitter.node_reset
#splitter.node_impurity
#splitter.node_split
#splitter.node_value

import ivy


def init_split(split_record, start_pos):
    split_record.impurity_left = float('inf')
    split_record.impurity_right = float('inf')
    split_record.pos = start_pos
    split_record.feature = 0
    split_record.threshold = 0.0
    split_record.improvement = float('-inf')
    split_record.missing_go_to_left = False
    split_record.n_missing = 0


class Splitter:
    def __init__(self, criterion, max_features, min_samples_leaf, min_weight_leaf, random_state):
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
        return (type(self), (self.criterion, self.max_features, self.min_samples_leaf,
                self.min_weight_leaf, self.random_state), self.__getstate__())

    def node_reset(self, start, end, weighted_n_node_samples):
        self.start = start
        self.end = end

        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end
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