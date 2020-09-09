from neural_net import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# functions
def n():
    return input('Ingrese un numero: ')

# input data
inputs = np.array([[0, 1, 0],
                   [0, 0, 0],0
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0], [0] ,[0], [0], [1], [1], [1]])

# create a neural network
NeuralN = NeuralNetwork(inputs, outputs)
NeuralN.train()

# create two new examples to test and predict
example1 = np.array([[1,1,0]])
example2 = np.array([[0,1,1]])
exampleUser = np.array([n(),n(),n()])

# print and predict the examples
print(NeuralN.predict(example1), '- Correct: ', example1[0][0])
print(NeuralN.predict(example2), '- Correct: ', example2[0][0])
print(NeuralN.predict(exampleUser), '- Correct: ', exampleUser[0][0])

# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(NeuralN.iters_hist, NeuralN.costerror_hist)
plt.xlabel('Iters')
plt.ylabel('Cost Error')
plt.show()