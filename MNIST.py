from FeedForward import FeedForwardNetwork, train
from readMNIST import read_data

from matplotlib import pyplot as plt
import numpy as np

def evaluate_on_test(X, T):
    y_hats = network.forward(x=X)
    classifications = np.argmax(y_hats, axis=1)

    confusion_matrix = np.zeros((10, 10), dtype=int)
    correct, wrong = 0, 0
    for i in range(X.shape[0]):
        actual_class = np.argmax(T[i, :])
        predicted_class = classifications[i]
        confusion_matrix[predicted_class, actual_class] += 1
        if actual_class == predicted_class:
            correct += 1
        else:
            wrong += 1

    plt.imshow(confusion_matrix, cmap='gray')
    plt.title(f'{round(100*(correct/(correct + wrong)), 2)}% success')
    plt.xlabel('target')
    plt.xticks(np.arange(10))
    plt.ylabel('classification')
    plt.yticks(np.arange(10))
    for i in range(10):
        for j in range(10):
            plt.text(i, j, f"{round(confusion_matrix[i, j])}", color='r')
    plt.show()

if __name__ == '__main__':
    train_images, train_labels, val_images, val_labels, test_images, test_labels = read_data(train_val_split=0.8)

    # Training settings
    epochs = 0
    batch_size = 64
    learning_rate = 0.001
    model_saving = {'save': {'do': False, 'path': '784_256_256_10_epoch3000.npy'},
                    'load': {'do': True, 'path': '784_256_256_10_epoch2500.npy'}}

    # Training network
    network = FeedForwardNetwork(neurons=[784, 256, 256, 10])

    if model_saving['load']['do']:
        network.load_model(path=model_saving['load']['path'])

    train(network, X=train_images, T=train_labels, epochs=epochs, lr=learning_rate, batch_size=batch_size, x_val=val_images, t_val=val_labels)

    if model_saving['save']['do']:
        network.save_model(path=model_saving['save']['path'])

    evaluate_on_test(X=test_images, T=test_labels)