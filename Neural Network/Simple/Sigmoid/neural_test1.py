from neural_net_sigmoid import NeuralNetwork
import numpy as np

neural_network = NeuralNetwork(3,1)
print('Random starting synaptic weights: ')
print(neural_network.synaptics_weights)

training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

neural_network.train(training_inputs, training_outputs, 20000)

print('Synaptics weights after training')
print(neural_network.synaptics_weights)

i = 1

while i < 5: 
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print('New situation: input data = ', A,B,C)
    print('Output data: ')
    print(neural_network.think(np.array([A,B,C])))

    input("Press Enter to continue..")
    i+=1
