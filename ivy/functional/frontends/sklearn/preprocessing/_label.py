from ivy.functional.frontends.sklearn.base import BaseEstimator, TransformerMixin
import ivy


class LabelEncoder(TransformerMixin, BaseEstimator):
    """
    A custom implementation of a label encoder for transforming categorical labels into numerical labels and vice versa.
    This class inherits from TransformerMixin and BaseEstimator classes from the sklearn.base module.

    Attributes:
    classes_ (list): A list of unique classes found in the data after fitting.
    class_to_index_ (list): A list mapping each unique class to an index.
    encoded_ (list): A list of encoded classes after transformation.
    decoded_ (list): A list of decoded classes after inverse transformation.
    """

    def __init__(self):
        """
        Initializes the LabelEncoder object.
        """
        self.classes_ = None
        self.class_to_index_ = None
        self.encoded_ = None
        self.decoded_ = None

    def fit(self, y):
        """
        Fits the encoder to the data.

        Args:
        y (array-like): The target values. It should be a 1D array or a 2D array with a single column.

        Returns:
        self: Returns the instance itself.
        """
        # Initialize the classes and class_to_index lists
        self.classes_= []
        self.class_to_index = []

        # Check the shape of the input data
        shape = y.shape
        if len(shape) == 2 and shape[1] == 1:
            # If the input data is a 2D array with a single column, reshape it to a 1D array
            y = y.reshape(-1)
        elif len(shape) != 1:
            # Raise an error if the input data is not a 1D array or a 2D array with a single column
            raise ValueError("y should be a 1D array, or a 2D array with a single column")

        # If the input data is an ivy.Array, convert it to a list
        if isinstance(y, ivy.Array):
            y = y.to_list()

        # Get the sorted list of unique classes in the input data
        self.classes_ = sorted(set(value for value in y if value is not None and value != ''))
        # Initialize the class_to_index list with the same length as the classes list
        self.class_to_index_ = [None] * len(self.classes_)
        # Map each unique class to an index
        for idx, value in enumerate(self.classes_):
            self.class_to_index_[idx] = value

        return self

    def fit_transform(self, y):
        """
        Fits the encoder to the data and then transforms the data.

        Args:
        y (array-like): The target values.

        Returns:
        self: Returns the instance itself.
        """
        return self.fit(y).transform(y)

    def transform(self, y):
        """
        Transforms the data into encoded form.

        Args:
        y (array-like): The target values.

        Returns:
        self: Returns the instance itself.
        """
        # Check if the encoder is fitted
        if self.classes_ is None or self.class_to_index_ is None:
            # Raise an error if the encoder is not fitted
            raise ValueError("This LabelEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # If the input data is an ivy.Array, convert it to a list
        if isinstance(y, ivy.Array):
            y = y.to_list()

        indices = []
        # For each class in the input data
        for cls in y:
            # If the class is in the class_to_index list
            if cls in self.class_to_index_:
                # Append the index of the class to the indices list
                indices.append(self.class_to_index_.index(cls))
            else:
                # Raise an error if the class is not in the class_to_index list
                raise ValueError(f"Class '{cls}' not found in class_to_index mapping. Ensure that the 'fit' method has been called with the correct data.")

        # Store the encoded classes
        self.encoded_ = indices
        return self

    def inverse_transform(self, indices):
        """
        Transforms the data back into original form.

        Args:
        indices (array-like): The encoded values.

        Returns:
        self: Returns the instance itself.
        """
        # Check if the encoder is fitted
        if self.classes_ is None:
            # Raise an error if the encoder is not fitted
            raise ValueError("This LabelEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # If the input data is an ivy.Array, convert it to a list
        if isinstance(indices, ivy.Array):
            indices = indices.to_list()

        # Get the original labels for the encoded values
        original_labels = [self.classes_[i] for i in indices]

        # Store the decoded classes
        self.decoded_ = original_labels
        return self