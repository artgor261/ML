from sklearn.base import RegressorMixin
import numpy as np

class SGDLinearRegressor(RegressorMixin):
    def __init__(self,
                 lr=0.01, regularization=1., delta_converged=1e-3, max_steps=1000,
                 batch_size=64):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None
        self.b = None

    def fit(self, X, Y):
        self.W = np.random.random(X.shape[1])
        self.b = np.random.random()
        self.data = np.concatenate((X, np.expand_dims(Y, axis=0).T), axis=1)
        self.steps = 0

        while self.steps < self.max_steps:
            np.random.shuffle(self.data)
            self.i = self.batch_size
            while self.i <= len(X):
                self.X_batch = self.data.T[:-1].T[self.i - self.batch_size : self.i]
                self.y_batch = self.data.T[-1][self.i - self.batch_size : self.i]
                self.f = np.dot(self.X_batch, self.W) + self.b
                self.err = self.y_batch - self.f
                self.grad_w = (2 * np.dot(self.X_batch.T, self.err) / self.batch_size) + self.regularization * self.W
                self.grad_b = 2 * np.sum(self.err) / self.batch_size
                if np.linalg.norm(self.W - (self.W - self.lr * self.grad_w)) < self.delta_converged:
                    return self
                self.W -= self.lr * self.grad_w
                self.b -= self.lr * self.grad_b
                self.i += self.batch_size
            self.steps += 1
        return self

    def predict(self, X):
        self.predicted = list()
        for i in X:
            self.predicted.append(np.dot(i, self.W) + self.b)
        return np.array(self.predicted)


model = SGDLinearRegressor()
model.fit(X_train, Y_train)

prediction = model.predict(X_test)
print(Y_test.shape, prediction.shape)
print("MAE : ", mean_absolute_error(Y_test, prediction))
print("Mean log : ", root_mean_squared_logarithmic_error(Y_test, prediction))
