import numpy as np
import matplotlib.pyplot as plt

class SimpleKNN:
    """
    Implementation of NN
    """
    def __init__(self, k=3, metric='euclidean', p=3):
        # k: Number of neighbors to consider.
        self.k = k
        
        # metric: The formula used to calculate.
        self.metric = metric
        
        # p: Only used for Minkowski distance. 
        self.p = p
        
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _calculate_distance(self, x1, x2):
        """
        Computes the distance between point x1 and point x2 based on the chosen metric.
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
            
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
            
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.abs(x1 - x2) ** self.p), 1 / self.p)
            
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict(self, X_new):
        """
        Predicts the class for a new point by looking at its neighbors.
        """
        # Step 1: Calculate distance from the new point to EVERY point in the training set.
        distances = []
        for i in range(len(self.X_train)):
            dist = self._calculate_distance(X_new, self.X_train[i])
            distances.append((dist, self.y_train[i], self.X_train[i]))

        # Step 2: Sort the list by distance (smallest to largest) and take the top 'k'.
        sorted_distances = sorted(distances, key=lambda x: x[0])[:self.k]

        # Step 3: Weighted Voting Mechanism
        # Sum weights based on 1/distance.
        class_votes = {}
        for dist, label, _ in sorted_distances:
            # Inverse distance weighting: 1 / (distance + small_epsilon)
            # Epsilon (1e-6) prevents division by zero if distance is 0.
            weight = 1 / (dist + 1e-6)
            
            # Sum the weights for this class
            class_votes[label] = class_votes.get(label, 0) + weight
        
        # Find the class with the highest total weight
        # max(dict, key=dict.get) gets the key with the largest value
        predicted_class = max(class_votes, key=class_votes.get)

        return predicted_class, sorted_distances

def generate_mock_data(n_samples=20):
    """
    Creates synthetic data for testing. 
    """
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

def visualize_knn(X, y, new_point, classifier, predicted_class, neighbors, ax=None):
    """
    Helper function to plot the data.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot training points (Class 0 = Blue, Class 1 = Red)
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], 
                color='blue', label='Class 0', alpha=0.6, s=50)
    
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], 
                color='red', label='Class 1', alpha=0.6, s=50)

    # Plot the new mystery point
    ax.scatter(new_point[0], new_point[1], 
                color='green', marker='*', s=150, label='New Point', edgecolors='black')

    # Draw lines connecting the mystery point to its K nearest neighbors
    for dist, label, coord in neighbors:
        line_color = 'blue' if label == 0 else 'red'
        
        # Calculate thickness based on weight (closer = thicker)
        thickness = (1 / (dist + 0.5)) * 2
        
        ax.plot([new_point[0], coord[0]], [new_point[1], coord[1]], 
                 'k--', alpha=0.4, linewidth=thickness)
        
        ax.scatter(coord[0], coord[1], s=80, facecolors='none', edgecolors=line_color, linewidth=1.5)

    metric_name = classifier.metric.capitalize()
    if classifier.metric == 'minkowski':
        metric_name += f' (p={classifier.p})'
    
    ax.set_title(f'{metric_name}, k={classifier.k}\nResult: Class {predicted_class}', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8)

def main():
    np.random.seed(42)
    print("Generating synthetic data...")
    X_train, y_train = generate_mock_data(n_samples=15)

    # Define the point we want to classify
    new_point = np.array([4.0, 4.0]) 

    k_values = [1, 3, 5]
    metrics = ['euclidean', 'manhattan', 'minkowski']
    
    fig, axes = plt.subplots(len(k_values), len(metrics), figsize=(12, 10))
    
    print("-" * 60)

    for row_idx, k in enumerate(k_values):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            # Create and train the model
            knn = SimpleKNN(k=k, metric=metric, p=3)
            knn.fit(X_train, y_train)
            
            # Predict
            prediction, neighbors = knn.predict(new_point)
            
            print(f"k={k} | {metric.ljust(10)} | Predicted: Class {prediction}")
            
            visualize_knn(X_train, y_train, new_point, knn, prediction, neighbors, ax=ax)
            
            if row_idx == len(k_values) - 1:
                ax.set_xlabel('Feature 1', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('Feature 2', fontsize=9)

    plt.suptitle(f'k vs metrics', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("-" * 60)
    plt.show()

if __name__ == "__main__":
    main()