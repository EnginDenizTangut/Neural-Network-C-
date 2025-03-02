# Neural Network Implementation in C++

## Overview
This project implements a simple feedforward neural network in C++ with two hidden layers. The network uses ReLU activation in the hidden layers and a sigmoid activation function in the output layer. It supports mini-batch gradient descent for training.

## Features
- Fully connected feedforward neural network
- Two hidden layers
- ReLU activation function for hidden layers
- Sigmoid activation function for output layer
- Mini-batch gradient descent
- Random weight initialization
- Backpropagation for training
- Network visualization

## Dependencies
- C++ Standard Library
- `<iostream>` for input/output operations
- `<vector>` for handling data structures
- `<cmath>` for mathematical functions
- `<thread>` and `<chrono>` for visualization delay
- `<random>` for weight initialization
- `<algorithm>` for shuffling training data

## Compilation & Execution
### Compilation
Use a C++ compiler such as `g++`:
```sh
 g++ -o neural_network neural_network.cpp
```
### Execution
Run the compiled program:
```sh
 ./neural_network
```

## Code Structure
1. **Weight Initialization**: Random initialization using a normal distribution.
2. **Activation Functions**: ReLU for hidden layers, Sigmoid for the output layer.
3. **Forward Propagation**: Computes the output of each layer.
4. **Backpropagation**: Updates weights using gradient descent.
5. **Mini-Batch Gradient Descent**: Trains the network in batches.
6. **Visualization**: Displays the network’s structure and activations.

## Training
- Uses a small dataset with predefined input values and target outputs.
- The network updates weights based on the mini-batch approach.
- Training runs for a specified number of epochs.

## Future Improvements
- Add support for multiple output neurons.
- Implement additional optimization techniques like Adam optimizer.
- Extend activation functions with Leaky ReLU or Softmax.
- Implement dataset loading from external files.

## License
This project is open-source and can be modified freely.

