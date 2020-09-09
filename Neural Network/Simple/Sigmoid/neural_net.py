import numpy as np

class NeuralNetwork:

    # Init variables and class
    def __init__(self, inputs, outputs):
        # Set the input and the output data
        self.inputs = inputs
        self.outputs = outputs
        # Set the random intialize weights
        np.random.seed(1)
        self.weights = 2 * np.random.random((3,1)) - 1
        # Set the costerror and iters history to test in plot
        self.costerror_hist = []
        self.iters_hist = []
    
    # Activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivate(self, x):
        return x * (1 - x)
    
    # Data will flow through the neural network
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights) + 1)
    
    def backpropagation(self):
        self.costerror = self.outputs - self.hidden
        deltaAdjusment = self.costerror * self.sigmoid_derivate(self.hidden)
        self.weights += np.dot(self.inputs.T, deltaAdjusment)
    
    def train(self, iters=1000):
        for iters in range(iters):
            # Execute the forwarde and produce the output
            self.feed_forward()
            # Execute the backpropagation and make correction based in the output
            self.backpropagation()
            # Keep track the error history over the iters(epoch list)
            self.costerror_hist.append(np.average(np.abs(self.costerror)))
            self.iters_hist.append(iters)
    
    def predict(self, new_input):
        new_input = new_input.astype(float)
        prediction = self.sigmoid(np.dot(new_input, self.weights) +  1)
        return prediction


    