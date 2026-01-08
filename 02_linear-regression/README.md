# Simple Linear Regression from Scratch

A basic implementation of a Linear Regression model using Gradient Descent and NumPy.

### How it works
* Model: Solving the linear equation y = mx + b.
* Optimization: Using Batch Gradient Descent to minimize Mean Squared Error (MSE).
* Update Rule: Adjusting slope (m) and intercept (b) based on calculated gradients.

### Requirements
* Python 3.x
* NumPy

### Usage
Run the script to train the model on sample data. The model will iterate 1000 times to find the line of best fit and then predict a value for x=10.

Example:
model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
model.train(X, y)
print(model.predict(10))