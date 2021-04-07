import numpy as np
from matplotlib import pyplot as plt

from make_data import make_data

def relu_diff(x):
    y = np.copy(x)
    y[y >= 0] = 1
    y[y < 0] = 0
    return y

def relu(x):
    return np.maximum(x, 0)

def train(network, X, T, epochs, lr, batch_size, x_val=np.array([]), t_val=np.array([])):
    train_loss, val_loss = [], []
    for i in range(epochs):
        for b in range(0, X.shape[0], batch_size):
            network.backward(x=X[b:b+batch_size], t=T[b:b+batch_size], batch_prop=batch_size/X.shape[0])
        network.step(lr=lr)
        train_loss.append(network.ABS_loss(x=X, t=T))
        if x_val.size:
            val_loss.append(network.ABS_loss(x=x_val, t=t_val))
        if i%10 == 9:
            plt.plot(np.arange(len(train_loss)), train_loss, label='Training loss')
            if val_loss:
                plt.plot(np.arange(len(val_loss)), val_loss, label='Validation loss')
            plt.grid()
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.show()
            print(f"Epoch {i+1}/{epochs} gives ABS_loss {round(train_loss[-1], 4)}")

class FeedForwardNetwork():
    def __init__(self, neurons):
        """
        :param n_neurons: array of neurons in each layer. First element is input layer, last node i output layer.
        """
        self.neurons = neurons
        self.n_layers = len(neurons)
        self.n_hidden_layers = len(neurons) - 2
        self.n_input = neurons[0]
        self.n_output = neurons[-1]

        # Initializing network
        self.initialize_weights()

    def initialize_weights(self):
        """
        Randomly initalizes weights and biases from a normal distribution scaled with sqrt(2/n) (from week 8 notes)

        For 2 -> 4 nodes, it is a [3x4] matrix, since the 3rd row denotes the bias. In the input we then just give a
        1 at this spot
        """
        self.w, self.Q = [], []
        for i in range(self.n_layers - 1):
            self.w.append(np.random.normal(loc=0, scale=np.sqrt(2/((self.neurons[i] + 1) * self.neurons[i+1])), size=(self.neurons[i] + 1, self.neurons[i+1])))
            self.Q.append(np.zeros((self.neurons[i] + 1, self.neurons[i+1])))

    def forward(self, x):
        """
        :param x: input of dimension [n, n_input]
        :return: output
        """
        self.z_s, self.h_s = [], []
        for w in self.w:
            self.h_s.append(x) # storing for backpropagation
            x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            x = x@w
            self.z_s.append(x) # storing for backpropagation
            x = relu(x) # ReLu transfer function

        y_hats = np.zeros((x.shape[0], self.w[-1].shape[-1]))
        for j in range(x.shape[0]):
            for h in range(self.w[-1].shape[-1]):
                y_hats[j, h] = np.exp(x[j, h]) / np.sum(np.exp(x[j, :]))

        return y_hats

    def backward(self, x, t, batch_prop):
        """
        :param x: input of dimension [n, n_input]
        :param t: target values of dimensions [n, n_output]
        :param batch_prop: proportion of data in batch

        Stores derivatives
        """
        deltas = [[]] * len(self.w)
        y = self.forward(x)
        deltas[-1] = np.subtract(y, t)

        # Calculating deltas
        for i in reversed(range(len(deltas)-1)):
            a_prime = relu_diff(self.z_s[i])
            deltas[i] = self.w[i+1][:-1, :].dot(deltas[i+1].T)
            deltas[i] = np.multiply(a_prime, deltas[i].T)

        # Calculating derivatives
        for i in range(len(deltas)):
            self.h_s[i] = np.concatenate((self.h_s[i], np.ones((self.h_s[i].shape[0], 1))), axis=1)
            self.Q[i] += ((self.h_s[i].T @ deltas[i]) / x.shape[0]) * batch_prop

    def step(self, lr):
        """
        :param lr: learning rate of how much we alter the weights and biases in each step

        Updates weights and biases
        """
        for i, (w, Q) in enumerate(zip(self.w, self.Q)):
            self.w[i] = w - Q*lr
        self.Q = [np.zeros((self.neurons[i] + 1, self.neurons[i+1])) for i in range(self.n_layers - 1)] # resets derivatives

    def ABS_loss(self, x, t):
        """
        :param x: input of dimension [n, n_input]
        :param t: target values of dimensions [n, n_output]
        :return: Absolute error loss
        """
        y = self.forward(x)
        loss = np.mean(np.abs(np.subtract(y, t)))
        return loss

    def save_model(self, path):
        """
        :param path: path for the model to be saved in

        Saves the parameters from self.w
        """
        np.save(path, self.w)

    def load_model(self, path):
        """
        :param path: path to model for loading

        Stores saved parameters in self.w
        """
        parameters = np.load(path, allow_pickle=True)
        self.w = parameters

if __name__ == '__main__':
    # Generating data
    n = 1000
    example_nr = 2
    noise = 1
    X, T, x, dim = make_data(example_nr, n, noise)

    # Centering around Origo
    X = X - np.mean(X, axis=0)

    # Training settings
    epochs = 100
    batch_size = 64
    learning_rate = 0.1
    model_saving = {'save': {'do': True, 'path': '2_10_10_2_epoch100.npy'},
                    'load': {'do': False, 'path': '2_10_10_2_epoch100.npy'}}

    # Training network
    network = FeedForwardNetwork(neurons=[2, 4, 6, 8, 2])

    if model_saving['load']['do']:
        network.load_model(path=model_saving['load']['path'])

    train(network, X=X, T=T, epochs=epochs, lr=learning_rate, batch_size=batch_size)

    if model_saving['save']['do']:
        network.save_model(path=model_saving['save']['path'])

    # Displaying results
    show_results = True
    if show_results:
        y_hat = network.forward(X)
        classes = np.zeros(X.shape[0], dtype=bool)
        classes[y_hat[:, 0] > 0.5] = True
        plt.scatter(X[classes, 0], X[classes, 1], label='True')
        plt.scatter(X[~classes, 0], X[~classes, 1], label='False')
        plt.show()
