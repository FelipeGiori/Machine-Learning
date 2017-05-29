import numpy as np
import supfunc as sf
from timeit import default_timer as timer


def main():

    # Reads data file
    print('Loading Data ...')
    X = np.genfromtxt('data_tp1', delimiter=',')
    [m, n] = X.shape
    print('Data Loaded ...')

    # Transfers column with label values to vector y
    y = np.array(X[:, 0])
    y = y.reshape(m, 1)

    # Sets first column to 1
    X[:, 0] = 1

    # Important variables
    init_layer_size = n - 1  # number of features
    hidden_layer_size = 25  # number of units on the hidden layer
    output_layer_size = 10  # number of classes

    # Creates theta vectors with randomly initialized weights
    theta1 = sf.initialize_weights(init_layer_size, hidden_layer_size)
    theta2 = sf.initialize_weights(hidden_layer_size, output_layer_size)

    start = timer()
    # Calls gradient function
    sf.gradient_descent(X, y, m, output_layer_size, theta1, theta2)
    sf.SGD(X, y, m, output_layer_size, theta1, theta2)
    sf.MBGD(X, y, m, output_layer_size, theta1, theta2)
    end = timer() - start
    print("The NN took %f seconds to be trained" % end)

if __name__ == '__main__':
    main()
