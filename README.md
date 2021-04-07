#MNIST through Feed Forward Network
#### s163918 - Eskild Børsting Sørensen

The implementation is purely done using numpy. The training process of the proposed network has been monitored
by observing the training vs validation error and no over-fitting is apparent. Thus, no regularization techniques 
have been used. Initialization is inspired from the lecture notes (week 9). Stochastic gradient descent is used 
by optimizing in the direction of the average negative gradient direction of all mini batches.

## Model description
- ReLU activation functions, Softmax output function
- Nodes: 784 - 256 - 256 - 10 with biases (269.322 Parameters)
- Batch size of 64
- learning rate of 0.1
- Train/validation-split = 0.8 (48.000/12.000)
- Trained for 2500 epochs

Requirements
- Numpy
- Matplotlib

## GUIDE

### MNIST

Re-create output results
1) Change working directory to this folder
2) Run MNIST.py without changing anything

To train networks
1) Change network parameters in MNIST.py
- Set epochs to be non-zero
- Set batch size
- Set learning rate
- Set number of layers and neurons as an array input to FeedForwardNetwork(...)
- (optional) load a pre-trained model
- choose a path to save the trained network
2) Change working directory to this folder
3) Run MNIST.py

To test networks (to get detection results for test images)
1) Change network parameters in MNIST.py
- Set epochs to 0
- choose loading path corresponding to the model
2) Change working directory to this folder
3) Run MNIST.py

### Toy data (2D)

Train a small model with 2 hidden layers and display the predicted results

1) Change working directory to this folder
2) Run FeedForward.py