from neural_net_sigmoid import NeuralNetwork
import numpy as np
import nnfs
from nnfs_dataset import spiral_data_generator

neural_network = NeuralNetwork(2,1)
print('Random starting synaptic weights: ')
print(neural_network.synaptics_weights)

np.random.seed(0)
nnfs.init()
def spiral_data(self,points, classes):
        X = np.zeros((points*classes, 2))
        y = np.zeros(points*classes, dtype='uint8')
        for class_number in range(classes):
            ix = range(points*class_number, points*(class_number+1))
            r = np.linspace(0.0, 1, points)  # radius
            t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
            X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            y[ix] = class_number
        return X, y
X, y = spiral_data(100, 3)

training_inputs = np.array([X])

training_outputs = np.array([y]).T

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