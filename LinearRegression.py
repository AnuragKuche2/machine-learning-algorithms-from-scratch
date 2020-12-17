import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


class LinearRegression:

    def __init__(self, lr=0.01, n_iters = 1000000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1/ n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

# b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
# m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


            # y_predicted = np.dot(X, self.weights) + self.bias

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


# from regression import LinearRegression
def mean_squared_error(y_true, predictions):
    return np.mean((y_true - predictions) ** 2)


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


regressor = LinearRegression(lr=0.015, n_iters=10000)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

