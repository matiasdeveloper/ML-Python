from neural_net_sigmoid import NeuralNetwork
import numpy as np

neural_network = NeuralNetwork(4,1)
print('Random starting synaptic weights: ')
print(neural_network.synaptics_weights)

training_inputs = np.array([[1.3,2,-3,1.7], [-1,1.2,-0.4,2], [-3,-1, 1, 1.4], [0,1.3,-2, 4]])
o = []

for i in training_inputs:
    e = sum(i)
    if e > 0:
        o.append(e)
        pass
    elif e <=0:
        o.append(0)
        pass

training_outputs = np.array([o]).T

print('Training outputs after training')
print(training_outputs)

neural_network.train(training_inputs, training_outputs, 20000)

print('Synaptics weights after training')
print(neural_network.synaptics_weights)

while True: 
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    D = str(input("Input 4: "))

    print('New situation: input data = ', A,B,C,D)
    print('Output data: ')
    print(neural_network.think(np.array([A,B,C,D])))

    if (int(input("Desea realizar otro calculo? 1=si | 2=no ->> ")) == 2):
        break
