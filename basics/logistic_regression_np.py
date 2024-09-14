import numpy as np
import time
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10000, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        """Add intercept to the feature matrix."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        """Compute the logistic loss."""
        epsilon = 1e-5  # To prevent log(0)
        return (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).mean()
    
    def fit(self, X, y):
        """Fit the logistic regression model."""
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # Initialize weights
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iterations):
            # Calculate predictions
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            # Compute the gradient
            gradient = np.dot(X.T, (h - y)) / y.size
            # Update the weights
            self.theta -= self.learning_rate * gradient
            
            if self.verbose and i % 1000 == 0:
                loss = self.__loss(h, y)
                print(f'Iteration {i}, loss: {loss}')
    
    def predict_prob(self, X):
        """Predict probability estimates for input data."""
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        """Predict binary labels for input data."""
        return self.predict_prob(X) >= threshold

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Initialize and train the model
    model = LogisticRegression(learning_rate=0.1, num_iterations=100, verbose=True)
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    # Make predictions
    preds = model.predict(X)
    print("Predictions:", preds)
    print("Actual labels:", y)
    print("execution time: ", end_time - start_time)