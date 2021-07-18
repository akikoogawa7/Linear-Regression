#%%
from sklearn import model_selection
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# split dataset into train and evaluation by 80 / 20
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.8)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.8)

# normalise
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean / X_std)

#%%
class DataLoader:
    def __init__(self, X, y, batch_size=16):
        self.batches = []
        idx = 0
        while idx < len(X):
            self.batches.append((X[idx : idx + 16], y[idx : idx + 16]))  # get batch
            idx += batch_size

    def __getitem__(self, idx):
        return self.batches[idx]  # X, y (in this case, 16 vectors)

train_loader = DataLoader(X, y)

#%%
class LinearRegression:
    def __init__(self, n_features): # initialises parameters
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

    def _get_mean_squared_error_loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)
    
    def fit(self, X, y, epochs=50):
        # controls scaling of step-size
        lr = 0.000001
        # initialise list of all costs
        all_costs = []
        for epoch in range(epochs):
            for X, y in train_loader.batches:
                # make predictions (y_hat)
                y_hat = self.predict(X)
                # compute loss and print
                loss = self._get_mean_squared_error_loss(y_hat, y)
                # compute the gradient of the loss with respect to model params
                grad_w, grad_b = self._compute_grads(X, y) # compute gradient for each weight and bias
                # update the params 
                self.w -= lr * grad_w # update weight
                self.b -= lr * grad_b # update bias
                all_costs.append(loss)
                print('loss:', loss)
        plot_loss(all_costs)
    
    def predict(self, X):
        return np.matmul(X, self.w) + self.b

    def _compute_grads(self, X, y):
        grads_for_individual_examples = []
        y_hat = self.predict(X)
        grad_b = 2 * np.mean(y_hat - y)
        for i in range(len(X)):
            grad_i = 2 * y_hat[i] - y[i] * X[i] # rate of change of loss with respect to weights for a particular example
            grads_for_individual_examples.append(grad_i)
        grad_w = np.mean(grads_for_individual_examples, axis=0)
        # the grad_w is what we will subtract from each of the Ws
        # each W is going to affect the predictions differently, which means they're going to affect the loss differently.
        return grad_w, grad_b
  
    def scalar_l1_norm(self):
        return np.linalg.norm(self.w, ord=1)

#%%
def plot_predictions(y_pred, y_true):
    samples = len(y_pred)
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()

def plot_loss(losses):
    """Helper function for plotting loss against epoch"""
    plt.figure() # make a figure
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(losses) # plot costs
    plt.show()

def plot_variance_bias(bias, variance):
    plt.figure()
    plt.ylabel()

#%%
linear_model = LinearRegression(n_features=X.shape[1]) # make predictions on data
linear_model.fit(X, y)
pred = linear_model.predict(X)
