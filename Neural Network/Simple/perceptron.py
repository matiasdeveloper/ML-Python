import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)
synaptics_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptics_weights)

for iter in range(20000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptics_weights))

    costerror = training_outputs - outputs
    adjustments = costerror * sigmoid_derivative(outputs)

    synaptics_weights += np.dot(input_layer.T, adjustments)

print('Synaptics weights after training')
print(synaptics_weights)

print('Output after training')
print(outputs)