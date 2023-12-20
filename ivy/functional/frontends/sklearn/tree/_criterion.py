from abc import ABC, abstractmethod
import ivy


class Criterion(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def reverse_reset(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, new_pos):
        raise NotImplementedError

    def proxy_impurity_improvement(self):
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
        return (self.weighted_n_node_samples / self.weighted_n_samples) * (
            impurity_parent
            - (self.weighted_n_right / self.weighted_n_node_samples * impurity_right)
            - (self.weighted_n_left / self.weighted_n_node_samples * impurity_left)
        )

    def node_value(self, dest, node_id):
        return dest


class ClassificationCriterion(Criterion):
    def __init__(self, n_outputs: int, n_classes: ivy.Array):
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
        max_n_classes = 0

        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]
            if n_classes[k] > max_n_classes:
                max_n_classes = n_classes[k]
        if isinstance(max_n_classes, ivy.Array):
            max_n_classes = ivy.to_scalar(max_n_classes)
        self.max_n_classes = max_n_classes

        self.sum_total = ivy.zeros((n_outputs, max_n_classes), dtype=ivy.float64)
        self.sum_left = ivy.zeros((n_outputs, max_n_classes), dtype=ivy.float64)
        self.sum_right = ivy.zeros((n_outputs, max_n_classes), dtype=ivy.float64)

    def init(
        self,
        y,
        sample_weight,
        weighted_n_samples,
        sample_indices,
        start,
        end,
    ):
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0
        w = 1.0

        for k in range(self.n_outputs):
            n_cls = ivy.to_scalar(self.n_classes[k])
            self.sum_total[k, :n_cls] = 0

        for p in range(start, end):
            i = sample_indices[p]
            if sample_weight is not None:
                w = sample_weight[i]
            for k in range(self.n_outputs):
                c = int(self.y[i, k])
                self.sum_total[k, c] += w

            self.weighted_n_node_samples += w

        self.reset()
        return 0

    def init_sum_missing(self):
        self.sum_missing = ivy.zeros(
            (self.n_outputs, self.max_n_classes), dtype=ivy.float64
        )

    def node_value(self, dest, node_id):
        for k in range(self.n_outputs):
            n_cls = ivy.to_scalar(self.n_classes[k])
            dest[node_id, k, :n_cls] = self.sum_total[k, :n_cls]
        return dest

    def init_missing(self, n_missing):
        w = 1.0
        self.n_missing = n_missing
        if n_missing == 0:
            return
        self.sum_missing[0 : self.n_outputs, 0 : self.max_n_classes] = 0
        self.weighted_n_missing = 0.0
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                c = int(self.y[i, k])
                self.sum_missing[k, c] += w

            self.weighted_n_missing += w

    def reset(self):
        self.pos = self.start
        (
            self.weighted_n_left,
            self.weighted_n_right,
            self.sum_left,
            self.sum_right,
        ) = _move_sums_classification(
            self,
            self.sum_left,
            self.sum_right,
            self.weighted_n_left,
            self.weighted_n_right,
            self.missing_go_to_left,
        )
        return 0

    def reverse_reset(self):
        self.pos = self.end
        (
            self.weighted_n_right,
            self.weighted_n_left,
            self.sum_right,
            self.sum_left,
        ) = _move_sums_classification(
            self,
            self.sum_right,
            self.sum_left,
            self.weighted_n_right,
            self.weighted_n_left,
            not self.missing_go_to_left,
        )
        return 0

    def update(self, new_pos):
        pos = self.pos
        end_non_missing = self.end - self.n_missing
        sample_indices = self.sample_indices
        sample_weight = self.sample_weight
        w = 1.0

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
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(ivy.to_scalar(self.n_classes[k])):
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]
        self.pos = new_pos
        return 0


class Gini(ClassificationCriterion):
    def node_impurity(self):
        gini = 0.0
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
        gini_left, gini_right = 0.0, 0.0
        for k in range(self.n_outputs):
            sq_count_left, sq_count_right = 0.0, 0.0
            for c in range(int(self.n_classes[k])):
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


# --- Helpers --- #
# --------------- #


def _move_sums_classification(
    criterion, sum_1, sum_2, weighted_n_1, weighted_n_2, put_missing_in_1
):
    for k in range(criterion.n_outputs):
        n = int(criterion.n_classes[k])
        sum_1[k, :n] = 0
        sum_2[k, :n] = criterion.sum_total[k, :n]

    weighted_n_1 = 0.0
    weighted_n_2 = criterion.weighted_n_node_samples

    return weighted_n_1, weighted_n_2, sum_1, sum_2
