import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


# constants
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 64
NUM_OUTPUT = 10



def cross_entropy_loss(y, yhat):
    """
    Function to calculate the cross entropy loss
    :param y: one-hot encoded labels
    :param yhat: predicted probabilities
    :return: cross entropy loss
    """
    return -(np.mean(np.sum(y*np.log(yhat), axis=1)))


def softmax(z):
    """
    Function to calculate softmax
    :param z: input to softmax
    :return: softmax output
    """
    return np.exp(z) / (np.sum(np.exp(z), axis=0, keepdims=True))  


def relu(m):
    """
    Function to apply the ReLU activation function to a matrix
    :param mat: matrix to apply ReLU to
    :return: ReLU(mat)
    """
    return np.maximum(m, 0)


def relu_der(input):
    """
    Function to calculate the derivative of the ReLU activation function
    :param input: input to the ReLU function
    :return: derivative of ReLU(input)
    """
    return (input > 0).astype(float)

def unpack (weightsandbiases):
    """
    Function to unpack a list of weights and biases into their individual np.arrays.
    :param weightsandbiases: list of weights and biases
    :return: list of weight matrices and list of bias vectors
    """
    # unpack arguments
    ws = []

    # weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    w = weightsandbiases[start:end]
    ws.append(w)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        w = weightsandbiases[start:end]
        ws.append(w)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    w = weightsandbiases[start:end]
    ws.append(w)

    # Reshape the weight "vectors" into proper matrices
    ws[0] = ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # convert from vectors into matrices
        ws[i] = ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    ws[-1] = ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsandbiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsandbiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsandbiases[start:end]
    bs.append(b)

    return ws, bs

def forward_prop(X, Y, weightsAndBiases, hidden_layers):
    """
    Perform forward propagation through the neural network
    :param X: input data
    :param Y: target labels
    :param weightsAndBiases: list of weights and biases
    :param hidden_layers: Integer specifying the number of units in each hidden layer
    :return: loss, pre-activations, post-activations, and predictions
    """

    weights, biases = unpack(weightsAndBiases)

    # Forward propagate through hidden layers
    activations = X
    pre_activations = []
    post_activations = []
    for layer in range(hidden_layers):
        weights_layer = weights[layer]
        biases_layer = biases[layer].reshape(-1, 1)

        pre_activations_layer = np.dot(weights_layer, activations) + biases_layer
        post_activations_layer = relu(pre_activations_layer)

        activations = post_activations_layer
        pre_activations.append(pre_activations_layer)
        post_activations.append(post_activations_layer)

    # Forward propagate through output layer
    weights_output = weights[-1]
    biases_output = biases[-1].reshape(-1, 1)
    pre_activations_output = np.dot(weights_output, activations) + biases_output
    post_activations_output = softmax(pre_activations_output)

    # Compute loss
    loss = (-1 / Y.shape[1]) * np.sum(np.log(post_activations_output) * Y)

    return loss, pre_activations + [pre_activations_output], post_activations + [post_activations_output], post_activations_output


import scipy.optimize

def back_prop(X, Y, weightsAndBiases, hidden_layers):
    """
    Perform backpropagation to compute gradients w.r.t. weights and biases
    :param X: input data
    :param Y: labels
    :param weightsAndBiases: list of weights and biases
    :param hidden_layers: number of hidden units in each layer
    :return: list of gradients w.r.t. weights and biases
    """
    _, h_z, h_h, yhat = forward_prop(X, Y, weightsAndBiases, hidden_layers)
    dJ_dWs = []
    dJ_dbs = []

    Ws, _ = unpack(weightsAndBiases)
    G = yhat - Y

    for i in range(hidden_layers, -1, -1):
        if i != hidden_layers:
            dh_dzs = relu_der(h_z[i])
            G = dh_dzs * G

        dj_db_term = np.sum(G, axis=1) / Y.shape[1]
        dJ_dbs.append(dj_db_term)

        if i == 0:
            dJ_dW_term = np.dot(G, X.T) / Y.shape[1]
        else:
            dJ_dW_term = np.dot(G, h_h[i - 1].T) / Y.shape[1]
        dJ_dWs.append(dJ_dW_term)

        G = np.dot(Ws[i].T, G)

    dJ_dWs.reverse()
    dJ_dbs.reverse()
    return np.hstack([dJ_dW.flatten() for dJ_dW in dJ_dWs] + [dJ_db.flatten() for dJ_db in dJ_dbs])


def update_w_b(W, B, gradW, gradB, epsilon, alpha, trainY):
    """
    Function to update the weights and biases
    :param W: list of weight matrices
    :param B: list of bias vectors
    :param gradW: list of gradients w.r.t. weights
    :param gradB: list of gradients w.r.t. biases
    :param epsilon: learning rate
    :param alpha: momentum
    :param trainY: training labels
    :return: updated weights and biases
    """
    for i in range(len(W)):
        W[i] = W[i] - (epsilon * gradW[i]) + (alpha * W[i] / trainY.shape[1])
        B[i] = B[i] - (epsilon * gradB[i])
    return W, B


def train(train_X, train_Y, weights_and_biases, hidden_layers, learning_rate, momentum):
    """
    Train the neural network using backpropagation and stochastic gradient descent
    :param train_X: training data
    :param train_Y: training labels
    :param weights_and_biases: list of weights and biases
    :param hidden_layers: number of hidden units in each layer
    :param learning_rate: learning rate for gradient updates
    :param momentum: momentum for gradient updates
    :return: updated list of weights and biases
    """

    # Compute the gradients using backpropagation
    gradients = back_prop(train_X, train_Y, weights_and_biases, hidden_layers)
    grad_W, grad_B = unpack(gradients)

    # Update the weights and biases using stochastic gradient descent with momentum
    weights, biases = unpack(weights_and_biases)
    weights, biases = update_w_b(weights, biases, grad_W, grad_B, learning_rate, momentum, train_Y)

    # Flatten and concatenate the weights and biases into a single array
    weights_and_biases = np.hstack([w.flatten() for w in weights] + [b.flatten() for b in biases])

    return weights_and_biases


def sgd(train_X, train_Y, epochs, batch_size, weightsAndBiases, H_Layers, learning_rate, alpha, valid_X, valid_Y):
    """
    Function to perform stochastic gradient descent
    :param train_X: training data
    :param train_Y: training labels
    :param epochs: number of epochs
    :param batch_size: batch size
    :param weightsAndBiases: list of weights and biases
    :param NUM_HIDDEN: number of hidden units
    :param H_Layers: number of hidden layers
    :param learning_rate: learning rate
    :param alpha: momentum
    :param valid_X: validation data
    :param valid_Y: validation labels
    :return: updated weights and biases
    """
   
    for _ in range(epochs):
        N_batches = int((len(train_X.T) / batch_size))
        init = 0
        end = batch_size
        for i in range(N_batches):
            mini_batch = train_X[:, init:end]

            y_mini_batch = train_Y[:, init:end]

            weightsAndBiases = train(mini_batch, y_mini_batch, weightsAndBiases,H_Layers,
                                                 learning_rate, alpha)
            init = end
            end = end + batch_size
           
        loss, _, _, yhat = forward_prop(valid_X, valid_Y, weightsAndBiases, H_Layers)
        acc = accuracy(yhat, valid_Y)
        print("Loss: ", loss, "Accuracy: ", acc)

    return weightsAndBiases



def accuracy(yhat, y):
    """
    Function to calculate accuracy
    :param yhat: predictions
    :param y: labels
    :return: accuracy
    """
    yhat = yhat.T
    y = y.T
    Yhat = np.argmax(yhat, 1)
    Y = np.argmax(y, 1)
    accuracy = 100 * np.sum(Y == Yhat) / y.shape[0]
    return accuracy


from itertools import product

def find_best_hyperparameters(trainX, trainY, testX, testY):
    """
    Function to find the best hyperparameters
    :param trainX: training data
    :param trainY: training labels
    :param testX: test data
    :param testY: test labels
    :return: best hyperparameters
    """
    # Define hyperparameters
    hidden_layers_list = [3]
    NUM_HIDDENbers_list = [64]
    mini_batch_size_list = [64]
    epsilon_list = [0.05]
    epochs_list = [150]
    alpha_list = [0.0002]

    # Preprocess training data
    change_order_index = np.random.permutation(trainX.shape[1])
    trainX = trainX[:, change_order_index]
    trainY = trainY[:, change_order_index]
    index_values = np.random.permutation(trainX.shape[1])
    train_X = trainX[:, index_values[:int(trainX.shape[1] * 0.8)]]
    valid_X = trainX[:, index_values[int(trainX.shape[1] * 0.8):]]
    train_Y = trainY[:, index_values[:int(trainX.shape[1] * 0.8)]]
    valid_Y = trainY[:, index_values[int(trainX.shape[1] * 0.8):]]

    # Initialize variables for early stopping
    best_loss = np.inf
    best_weightsAndBiases = None
    epochs_without_improvement = 0
    max_epochs_without_improvement = 3

    # Generate all combinations of hyperparameters
    hyperparameters_list = list(product(hidden_layers_list, NUM_HIDDENbers_list, mini_batch_size_list, epsilon_list, epochs_list, alpha_list))

    for h_layers, NUM_HIDDEN, batch_size, learning_rate, epochs, alpha in hyperparameters_list:
        # Initialize weights and biases
        weightsAndBiases = init_weights_and_biases()

        # Train the model with the current hyperparameters
        for epoch in range(epochs):
            print("Epoch: ", epoch )
            weightsAndBiases = sgd(train_X, train_Y, 1, batch_size, weightsAndBiases, h_layers, learning_rate, alpha, valid_X, valid_Y)

            # Compute loss and accuracy on validation set
            loss, _, _, yhat = forward_prop(valid_X, valid_Y, weightsAndBiases, h_layers)
            acc = accuracy(yhat, valid_Y)

            # Check for early stopping
            if loss < best_loss:
                best_loss = loss
                best_weightsAndBiases = weightsAndBiases
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= max_epochs_without_improvement:
                    print("Stopping early at epoch", epoch)
                    break

        # Compute accuracy on test set using the best weights and biases
        _, _, _, yhat_test = forward_prop(testX, testY, best_weightsAndBiases, h_layers)
        acc_test = accuracy(yhat_test, testY)

        # Print results for current hyperparameters
        print("Hyperparameters:", (h_layers, NUM_HIDDEN, batch_size, learning_rate, epochs, alpha))
        print("Validation loss:", loss)
        print("Validation accuracy:", acc)
        print("Test accuracy:", acc_test)
        print()

    # Return best hyperparameters and weights and biases
    return best_loss, best_weightsAndBiases


def show_W0 (W):
    """
    Show the first layer of the network
    param W: the weights and biases
    return: None
    """
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()


def init_weights_and_biases():
    """
    Initialize the weights and biases
    param NUM_HIDDEN: number of hidden neurons
    param H_Layers: number of hidden layers
    return: weights and biases
    """
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])



if __name__ == "__main__":
  # Load and preprocess the data
    train_X = np.reshape(np.load("/content/drive/MyDrive/HW4-dl/fashion_mnist_train_images.npy"), (-1, 28 * 28)) / 255
    train_Y = np.load("/content/drive/MyDrive/HW4-dl/fashion_mnist_train_labels.npy")
    test_X = np.reshape(np.load("/content/drive/MyDrive/HW4-dl/fashion_mnist_test_images.npy"), (-1, 28 * 28)) / 255
    test_Y = np.load("/content/drive/MyDrive/HW4-dl/fashion_mnist_test_labels.npy")

    # One-hot encode the labels
    train_Y = np.eye(train_Y.max() + 1)[train_Y]
    test_Y = np.eye(test_Y.max() + 1)[test_Y]

    # Transpose the input arrays to match the expected format
    train_X = train_X.T
    train_Y = train_Y.T
    test_X = test_X.T
    test_Y = test_Y.T

    # Initialize the weights and biases
    weightsAndBiases = init_weights_and_biases()

    # Check the gradient to ensure it's correct
    grad = scipy.optimize.check_grad(
        lambda wab: forward_prop(train_X[:, :5], train_Y[:, :5], wab, NUM_HIDDEN_LAYERS)[0],
        lambda wab: back_prop(train_X[:, :5], train_Y[:, :5], wab, NUM_HIDDEN_LAYERS),
        weightsAndBiases)
    print(f"Gradient check: {grad}\n")

    # Find the best hyperparameters for the model
    best_loss, best_weights = find_best_hyperparameters(train_X, train_Y, test_X, test_Y)

    show_W0(best_weights)