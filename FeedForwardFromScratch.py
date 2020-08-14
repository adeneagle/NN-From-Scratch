import numpy as np
import matplotlib.pyplot as plt
from ActivationsAndLosses import *

class NeuralNet:
    def __init__(self, learning_rate, loss):
        self.layers = []
        self.learning_rate = learning_rate
    def predict(self, output):
        predictions = []
        out = self.layers[0].forward(output)
        predictions.append(out)
        for x in range(len(self.layers)-1):
            index = x+1
            prediction = self.layers[index].forward(predictions[-1])
            predictions.append(prediction)
        return predictions[-1]
    def add(self, layer):
        self.layers.append(layer)
    def fit(self, X, Y, epochs):
        for p in range(epochs):
            for x, y in zip(X, Y):
                predictions = self.predict(x)
                error = np.sum(getLoss(predictions, y, loss))
                if p % 100 == 0:
                    print(f"Error: {error}")
                
            

class FeedForwardLayer:
    def __init__(self, n_inputs, n_nodes, activation, loss):
        self.n_nodes = n_nodes
        self.errors = []
        self.loss = loss
        self.activation = activation
        self.n_inputs = n_inputs
        self.weights = np.random.rand(n_inputs, n_nodes)
        self.bias = np.zeros((1, n_nodes))
    def print_weights(self):
        print(self.weights)
        print(self.bias)
    def forward(self, x):
        result = np.dot(x, self.weights) + self.bias
        return getActivation(result, self.activation)
    def backward_propagation(self, X, Y, learning_rate, epochs):
        for p in range(epochs):
            error = np.sum(getLoss(self.forward(X[0]), Y[0]), self.loss)
            self.errors.append(error)
            if p % 100 == 0:
                print(f"Error: {error}")
            for v in range(len(X)):
                x = X[v]
                y = Y[v]
                m = np.dot(x, self.weights) + self.bias
                z = self.forward(x)
                
                dE_dz = getLoss(z, y, self.activation, derivative=True)
                dz_dy = getActivation(m, self.activation, derivative=True)
                
                dE_dy = dE_dz*dz_dy
                weight_update = np.dot(np.transpose([x]), dE_dy)
                self.weights = self.weights - learning_rate*weight_update
                self.bias = self.bias - learning_rate*dE_dy
    def plot_error(self):
        plt.plot(self.errors)




X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.float64)
Y = np.array([[1], [1], [0], [0]], dtype=np.float64)

layer1 = FeedForwardLayer(2, 10, "sigmoid", "MSE")
layer2 = FeedForwardLayer(10, 10, "sigmoid", "MSE")
layer3 = FeedForwardLayer(10, 1, "sigmoid", "MSE")

model = NeuralNet(0.01, "MSE")
model.add(layer1)
model.add(layer2)
model.add(layer3)