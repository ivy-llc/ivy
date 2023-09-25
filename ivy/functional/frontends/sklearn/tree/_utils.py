from ..utils._random import our_rand_r

RAND_R_MAX = 2147483647

# =============================================================================
# Helper functions
# =============================================================================


def rand_int(low, high, random_state):
    """Generate a random integer in [low; high)."""
    #return low + our_rand_r(random_state) % (high - low)
    return low + 1 % (high - low) #TODO: FIX THE RANDOM NUMBER ISSUE


class WeightedMedianCalculator:
    def __init__(self, initial_capacity):
        self.initial_capacity = initial_capacity
        self.samples = []
        self.total_weight = 0
        self.k = 0
        self.sum_w_0_k = 0

    def size(self):
        return len(self.samples)

    def reset(self):
        self.samples = []
        self.total_weight = 0
        self.k = 0
        self.sum_w_0_k = 0

    def push(self, data, weight):
        original_median = self.get_median() if self.size() != 0 else 0.0
        self.samples.append((data, weight))
        self.update_median_parameters_post_push(data, weight, original_median)

    def update_median_parameters_post_push(self, data, weight, original_median):
        if self.size() == 1:
            self.k = 1
            self.total_weight = weight
            self.sum_w_0_k = self.total_weight
            return

        self.total_weight += weight

        if data < original_median:
            self.k += 1
            self.sum_w_0_k += weight

            while (
                self.k > 1
                and (self.sum_w_0_k - self.samples[self.k - 1][1])
                >= self.total_weight / 2.0
            ):
                self.k -= 1
                self.sum_w_0_k -= self.samples[self.k][1]

    def remove(self, data, weight):
        original_median = self.get_median() if self.size() != 0 else 0.0
        self.samples = [(d, w) for (d, w) in self.samples if d != data or w != weight]
        self.update_median_parameters_post_remove(data, weight, original_median)

    def pop(self):
        original_median = self.get_median() if self.size() != 0 else 0.0
        if self.size() == 0:
            return None, None
        data, weight = self.samples.pop(0)
        self.update_median_parameters_post_remove(data, weight, original_median)
        return data, weight

    def update_median_parameters_post_remove(self, data, weight, original_median):
        if not self.samples:
            self.k = 0
            self.total_weight = 0
            self.sum_w_0_k = 0
            return

        if self.size() == 1:
            self.k = 1
            self.total_weight -= weight
            self.sum_w_0_k = self.total_weight
            return

        self.total_weight -= weight

        if data < original_median:
            self.k -= 1
            self.sum_w_0_k -= weight

            while self.k < self.size() and self.sum_w_0_k < self.total_weight / 2.0:
                self.k += 1
                self.sum_w_0_k += self.samples[self.k - 1][1]

    def get_median(self):
        if self.sum_w_0_k == (self.total_weight / 2.0):
            return (self.samples[self.k][0] + self.samples[self.k - 1][0]) / 2.0
        if self.sum_w_0_k > (self.total_weight / 2.0):
            return self.samples[self.k - 1][0]
