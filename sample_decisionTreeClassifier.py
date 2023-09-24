import time
import numpy as np
from ivy.functional.frontends.sklearn.tree import DecisionTreeClassifier

num_classes = 3
X = np.random.rand(10, 5)
y = np.random.randint(num_classes, size=10)
# Train a scikit-learn DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Measure inference time without Hummingbird
start_time = time.time()
y_pred_sklearn = clf.predict(X)
end_time = time.time()
sklearn_inference_time = end_time - start_time
