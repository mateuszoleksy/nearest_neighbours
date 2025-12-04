import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import warnings
from collections import Counter

# Suppress runtime warnings from NumPy about invalid values (e.g., division by zero handled by 1e-6)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# 1. USER'S SimpleKNN IMPLEMENTATION (Modified to be standalone)
# ==============================================================================

class SimpleKNN:
    """
    Implementation of KNN using brute-force distance calculation.
    Uses inverse distance weighting by default.
    """
    def __init__(self, k=3, metric='euclidean', p=3):
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN is a 'lazy' learner, so fit just stores the data.
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _calculate_distance(self, x1, x2):
        """Computes distance based on the chosen metric."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            # This is correct for the original code's implementation:
            return np.power(np.sum(np.abs(x1 - x2) ** self.p), 1 / self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict(self, X_new_data):
        """Predicts the class for an array of new points."""
        predictions = []
        for X_new in X_new_data:
            distances = []
            for i in range(len(self.X_train)):
                dist = self._calculate_distance(X_new, self.X_train[i])
                distances.append((dist, self.y_train[i]))

            # Step 2: Sort and take the top 'k'.
            sorted_distances = sorted(distances, key=lambda x: x[0])[:self.k]

            # Step 3: Inverse Distance Weighted Voting
            class_votes = {}
            for dist, label in sorted_distances:
                # Epsilon (1e-6) prevents division by zero if distance is 0.
                weight = 1 / (dist + 1e-6)
                class_votes[label] = class_votes.get(label, 0) + weight
            
            # Find the class with the highest total weight
            if not class_votes:
                 # Should not happen with k>=1, but a safe fallback
                 predicted_class = self.y_train[0]
            else:
                 predicted_class = max(class_votes, key=class_votes.get)

            predictions.append(predicted_class)
            
        return np.array(predictions)

# ==============================================================================
# 2. DATA GENERATION AND COMPARISON LOGIC
# ==============================================================================

def generate_mock_data(n_samples=100):
    """Creates synthetic data for testing, similar to the original."""
    np.random.seed(42)
    # Class 0: A cluster centered around coordinate (2, 2)
    class_0_x = np.random.normal(2, 1, n_samples)
    class_0_y = np.random.normal(2, 1, n_samples)
    class_0 = np.column_stack((class_0_x, class_0_y))
    labels_0 = np.zeros(n_samples, dtype=int)

    # Class 1: A cluster centered around coordinate (6, 6)
    class_1_x = np.random.normal(6, 1, n_samples)
    class_1_y = np.random.normal(6, 1, n_samples)
    class_1 = np.column_stack((class_1_x, class_1_y))
    labels_1 = np.ones(n_samples, dtype=int)

    X = np.vstack((class_0, class_1))
    y = np.concatenate((labels_0, labels_1))
    
    return X, y

def run_comparison(X_train, y_train, X_test, y_test, k, metric, p_val, simple_knn_class=SimpleKNN):
    """
    Runs both SimpleKNN and scikit-learn's KNN for comparison.
    """
    print(f"\n--- Testing k={k}, Metric='{metric}' (p={p_val}) ---")
    
    # --- 1. SimpleKNN (Inverse Distance Weighting is fixed) ---
    start_time = time.time()
    simple_knn = simple_knn_class(k=k, metric=metric, p=p_val)
    simple_knn.fit(X_train, y_train)
    simple_predictions = simple_knn.predict(X_test)
    simple_time = time.time() - start_time
    simple_acc = accuracy_score(y_test, simple_predictions)
    
    print(f"SimpleKNN (Inverse Distance):")
    print(f"  Time: {simple_time:.4f}s")
    print(f"  Accuracy: {simple_acc:.4f}")
    
    # --- 2. Scikit-learn (Inverse Distance Weighting) ---
    # Equivalent to SimpleKNN's logic
    start_time = time.time()
    # Note: Scikit-learn uses 'minkowski' by default, where p=2 is Euclidean.
    # We must explicitly use the correct metric name for scikit-learn.
    
    # Map metrics for sklearn: 'euclidean' and 'manhattan' are names, 'minkowski' is also a name.
    # Scikit-learn's 'euclidean' is equivalent to 'minkowski' with p=2
    # Scikit-learn's 'manhattan' is equivalent to 'minkowski' with p=1
    
    sklearn_metric = metric
    if metric == 'euclidean' and p_val == 2:
        # Use 'minkowski' p=2 for strict equivalence
        sklearn_metric = 'minkowski'
    elif metric == 'manhattan' and p_val == 1:
        # Use 'minkowski' p=1 for strict equivalence
        sklearn_metric = 'minkowski'
        
    sklearn_knn_dist = KNeighborsClassifier(
        n_neighbors=k, 
        p=p_val, 
        metric=sklearn_metric, 
        weights='distance' # This matches SimpleKNN's inverse distance logic
    )
    sklearn_knn_dist.fit(X_train, y_train)
    sklearn_predictions_dist = sklearn_knn_dist.predict(X_test)
    sklearn_time_dist = time.time() - start_time
    sklearn_acc_dist = accuracy_score(y_test, sklearn_predictions_dist)
    
    # Check if predictions are identical for direct logic comparison
    consistency = np.array_equal(simple_predictions, sklearn_predictions_dist)
    
    print(f"Scikit-learn (Inverse Distance):")
    print(f"  Time: {sklearn_time_dist:.4f}s")
    print(f"  Accuracy: {sklearn_acc_dist:.4f}")
    print(f"  Result Consistency with SimpleKNN: {'MATCH' if consistency else 'MISMATCH'}")

    # --- 3. Scikit-learn (Uniform Weighting - The default/standard method) ---
    start_time = time.time()
    sklearn_knn_unif = KNeighborsClassifier(
        n_neighbors=k, 
        p=p_val, 
        metric=sklearn_metric, 
        weights='uniform' # Scikit-learn's default
    )
    sklearn_knn_unif.fit(X_train, y_train)
    sklearn_predictions_unif = sklearn_knn_unif.predict(X_test)
    sklearn_time_unif = time.time() - start_time
    sklearn_acc_unif = accuracy_score(y_test, sklearn_predictions_unif)

    print(f"Scikit-learn (Uniform Weighting - Standard Default):")
    print(f"  Time: {sklearn_time_unif:.4f}s")
    print(f"  Accuracy: {sklearn_acc_unif:.4f}")


def main_tests():
    # --- Setup ---
    print("--- Generating Standard Test Data (100 training/100 test samples) ---")
    X, y = generate_mock_data(n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    print(f"Data split: {len(X_train)} training points, {len(X_test)} test points.")
    
    print("\n=======================================================")
    print("== TEST SCENARIOS: LOGIC AND ACCURACY COMPARISON ==")
    print("=======================================================")

    # Test Case 1: Euclidean Distance, k=3
    # Note: SimpleKNN default p=3 is ignored here because metric='euclidean' logic is used.
    run_comparison(X_train, y_train, X_test, y_test, k=3, metric='euclidean', p_val=2)

    # Test Case 2: Manhattan Distance, k=5
    run_comparison(X_train, y_train, X_test, y_test, k=5, metric='manhattan', p_val=1)

    # Test Case 3: Minkowski Distance, k=1
    # SimpleKNN's default p=3 is used here. Scikit-learn matches it.
    run_comparison(X_train, y_train, X_test, y_test, k=1, metric='minkowski', p_val=3)

    print("\n=======================================================")
    print("== TEST SCENARIO: PERFORMANCE COMPARISON (SPEED) ==")
    print("=======================================================")
    
    # Performance Test with a larger dataset
    N_PERFORMANCE = 2000
    print(f"--- Generating LARGE Test Data ({2 * N_PERFORMANCE} total samples) ---")
    X_perf, y_perf = generate_mock_data(n_samples=N_PERFORMANCE)
    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_perf, y_perf, test_size=0.5, random_state=42)
    print(f"Data split: {len(X_p_train)} training points, {len(X_p_test)} test points.")

    # Run performance test with standard Euclidean (L2/Minkowski p=2) and k=5
    K_PERF = 5
    P_PERF = 2

    # --- SimpleKNN Performance (Brute-Force) ---
    start_time = time.time()
    simple_knn_perf = SimpleKNN(k=K_PERF, metric='euclidean', p=P_PERF)
    simple_knn_perf.fit(X_p_train, y_p_train)
    _ = simple_knn_perf.predict(X_p_test)
    simple_time_perf = time.time() - start_time
    
    # --- Scikit-learn Performance (Optimized) ---
    start_time = time.time()
    sklearn_knn_perf = KNeighborsClassifier(
        n_neighbors=K_PERF, 
        p=P_PERF, 
        metric='minkowski', 
        weights='distance', 
        algorithm='auto' # Uses BallTree/KDTree/Brute based on data
    )
    sklearn_knn_perf.fit(X_p_train, y_p_train)
    _ = sklearn_knn_perf.predict(X_p_test)
    sklearn_time_perf = time.time() - start_time

    print(f"\n--- Performance Results (k={K_PERF}, Euclidean) ---")
    print(f"SimpleKNN Time (Brute-Force): {simple_time_perf:.4f}s")
    print(f"Scikit-learn Time (Optimized): {sklearn_time_perf:.4f}s")
    print(f"Speedup Factor: {simple_time_perf / sklearn_time_perf:.1f}x (Scikit-learn is faster)")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    main_tests()