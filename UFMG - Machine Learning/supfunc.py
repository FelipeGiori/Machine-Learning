import numpy as np


# Prints useful information about feature matrix X
def xinfo(X):
    print("X info:")
    print("X.Shape: " + repr(X.shape))
    print("First Values: " + repr(X))


# Prints useful information about label vector Y
def yinfo(y):
    print("Y info:")
    print("Y.Shape: " + repr(y.shape))
    print("First Values: " + repr(y[:20]))


# Prints useful information about weight vector theta
def thetainfo(theta):
    print("Theta info:")
    print("theta.Shape: " + repr(theta.shape))
    print("First Values: " + repr(theta))


# Back propagation function
def back_propagation(X, y, m, k, theta1, theta2):
    # Creates y's eye matrix
    y_matrix = eyematrix(y, m, k)
    cost = 0

    # Creates matrices do store gradients for back propagation
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    # Performs feature scaling
    X = feature_scaling(X)

    # Loops through the cases
    for i in range(m):
        # Loads the i-th case to a1
        a1 = X[i, :]

        # Calculates cost of hidden layer and appends bias' multiplier
        a2 = sigmoid(theta1.dot(a1))
        a2 = np.append(np.ones((1, 1)), a2)

        # Calculates cost of output layer
        a3 = sigmoid(a2.dot(theta2.T))

        # Calculates the cost for the i-th case
        cost += np.sum(-y_matrix[i, :] * np.log(a3) - (1.0 - y_matrix[i, :]) * np.log(1.0 - a3))

        # Calculates the error and the gradients
        delta3 = a3 - y_matrix[i, :]
        theta2_grad += np.outer(delta3, a2)
        delta2 = theta2.T.dot(delta3) * a2 * (1 - a2)
        theta1_grad += np.outer(delta2[1:], a1)

    cost = cost / m
    theta1_grad = theta1_grad / m
    theta2_grad = theta2_grad / m

    # Prints the error
    print_error(theta1, theta2, X, y)
    # print(cost)

    return theta1_grad, theta2_grad


# Gradient descent
def gradient_descent(X, y, m, k, theta1, theta2):
    num_iter = 1500
    r = 0.5
    for i in range(num_iter):
        [theta1_grad, theta2_grad] = back_propagation(X, y, m, k, theta1, theta2)
        theta1 = theta1 - r * theta1_grad
        theta2 = theta2 - r * theta2_grad


# Creates eye matrix
def eyematrix(y, m, k):
    y_matrix = np.zeros((m, k))
    for i in range(y.size):
        y_matrix[i][np.int(y[i, 0])] = 1
    return y_matrix


# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Feature Scaling
def feature_scaling(X):
    return X/np.max(X)


# Checks the predicted result
def predict(theta1, theta2, X):
    m = X.shape[0]
    a2 = sigmoid(X.dot(theta1.T))
    a2 = np.append(np.ones((m, 1)), a2, axis=1)
    a3 = sigmoid(a2.dot(theta2.T))
    return np.argmax(a3, axis=1)


# Prints the error
def print_error(theta1, theta2, X, y):
    m = X.shape[0]
    pred = predict(theta1, theta2, X)
    pred = pred.reshape(m, 1)
    accuracy = 100 * np.mean(pred == y)
    print("Training error: %.2f" % (100-accuracy))


# Randomly initialize weights
def initialize_weights(i, j):
    eps = 0.12
    return np.random.uniform(-eps, eps, (j, 1 + i))


# Stochastic gradient descent
def SGD(X, y, m, k, theta1, theta2):
    # Performs feature scaling and shuffles it
    X = feature_scaling(X)

    # Creates y's eye matrix
    y_matrix = eyematrix(y, m, k)

    r = 0.5
    num_iter = 100

    for t in range(num_iter):
        for i in range(m):
            # Loads the i-th case to a1
            a1 = X[i, :]

            # Calculates cost of hidden layer and appends bias' multiplier
            a2 = sigmoid(theta1.dot(a1))
            a2 = np.append(np.ones((1, 1)), a2)

            # Calculates cost of output layer
            a3 = sigmoid(a2.dot(theta2.T))

            # Calculates the cost for the i-th case
            cost = np.sum(-y_matrix[i, :] * np.log(a3) - (1.0 - y_matrix[i, :]) * np.log(1.0 - a3))
            # print(cost)

            # Calculates the error and the gradients
            delta3 = a3 - y_matrix[i, :]
            theta2_grad = np.outer(delta3, a2)
            delta2 = theta2.T.dot(delta3) * a2 * (1 - a2)
            theta1_grad = np.outer(delta2[1:], a1)

            # updates the weights
            theta1 = theta1 - r * theta1_grad
            theta2 = theta2 - r * theta2_grad

            # Prints the error every 1000th iteration
            if(i % 1000 == 0):
                print_error(theta1, theta2, X, y)


# Mini-batch gradient descent
def MBGD(X, y, m, k, theta1, theta2):
    # Performs feature scaling and shuffles it
    X = feature_scaling(X)

    # Creates y's eye matrix
    y_matrix = eyematrix(y, m, k)

    r = 0.5
    num_iter = 100
    batch_size = 10
    cost = 0

    # Creates matrices do store gradients for back propagation
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    for t in range(num_iter):
        for i in range(m):
            # Loads the i-th case to a1
            a1 = X[i, :]

            # Calculates cost of hidden layer and appends bias' multiplier
            a2 = sigmoid(theta1.dot(a1))
            a2 = np.append(np.ones((1, 1)), a2)

            # Calculates cost of output layer
            a3 = sigmoid(a2.dot(theta2.T))

            # Calculates the cost for the i-th case
            cost += np.sum(-y_matrix[i, :] * np.log(a3) - (1.0 - y_matrix[i, :]) * np.log(1.0 - a3))

            # Calculates the error and the gradients
            delta3 = a3 - y_matrix[i, :]
            theta2_grad += np.outer(delta3, a2)
            delta2 = theta2.T.dot(delta3) * a2 * (1 - a2)
            theta1_grad += np.outer(delta2[1:], a1)

            # updates the weights
            if(i % batch_size == 0):
                theta1 = theta1 - r * (theta1_grad / batch_size)
                theta2 = theta2 - r * (theta2_grad / batch_size)
                theta1_grad = np.zeros(theta1.shape)
                theta2_grad = np.zeros(theta2.shape)

            # Prints the error every 1000th iteration
            if(i % 1000 == 0):
                print_error(theta1, theta2, X, y)
                # print(cost/1000)
                # cost = 0
