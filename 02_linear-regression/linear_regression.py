import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.slope = 0
        self.intercept = 0

    def train(self, X, y):
        """Standard Gradient Descent to find the best line."""
        n = len(X)
        
        for _ in range(self.iterations):
            # Prediction
            y_pred = (self.slope * X) + self.intercept
            
            # Gradients calculation
            d_slope = (-2/n) * sum(X * (y - y_pred))
            d_intercept = (-2/n) * sum(y - y_pred)
            
            # Updating weights
            self.slope -= self.lr * d_slope
            self.intercept -= self.lr * d_intercept

    def predict(self, X):
        return (self.slope * X) + self.intercept

# --- Testing the code ---
if __name__ == "__main__":
    # Sample data: y = 2x + 1
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 7, 9, 11])

    # Initialize and train
    model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
    model.train(X, y)

    # Print results
    print(f"Learned Slope: {model.slope:.2f}")
    print(f"Learned Intercept: {model.intercept:.2f}")
    
    # Try a prediction
    test_val = 10
    prediction = model.predict(test_val)
    print(f"Prediction for x={test_val}: {prediction:.2f}")
