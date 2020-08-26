import numpy as np

class NeuralNetwork():
    def __init__(self, n_inputs, n_neurons):
        np.random.seed(1)
        self.synaptics_weights = 2 * np.random.random((n_inputs,n_neurons)) - 1
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivate(self,x):
        return x * (1-x)
    
    def train(self, training_inputs, training_outputs, iters):
        for i in range(iters):
            output = self.think(training_inputs)
            costerror = training_outputs - output
            adjustment = np.dot(training_inputs.T, costerror * self.sigmoid_derivate(output))
            self.synaptics_weights += adjustment
    
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptics_weights) + 1)

        return output
