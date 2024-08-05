# Dylan Kasanders
# 202410_data571_kasandd
DATA 471/571 Winter 2024 (kasandd)


This project implements a deep neural network utilizing NumPy. This project supports:

- A standard single hidden layer neural network
- An arbitrarily deep neural network
- Both classification and regression output modes
- Minibatch training, including stochastic gradient descent

The command line arguments for prog.py are:

- **-v**: If used, the program will operate in verbose mode, printing the training set and dev set performance after each update
- **TRAIN_FEAT_FN**: The name of the training set feature file.
- **TRAIN_TARGET_FN**: The name of the training set target (label) file.
- **DEV_FEAT_FN**: The name of the development set feature file, in the same format as **TRAIN_FEAT_FN**.
- **DEV_TARGET_FN**: The name of the development set target (label) file, in the same format as **TRAIN_TARGET_FN**.
- **EPOCHS**: The total number of epochs (passes through the data) to train for.
- **LEARNRATE**: The step size to use for training.
- **NUM_HIDDEN_UNITS**: The dimension of the hidden layers. All hidden layers will have the same size.
- **PROBLEM_MODE**: This should either be **C** for classification tasks or **R** for regression tasks.
- **HIDDEN_UNIT_ACTIVATION**: This is the activation function for each hidden layer, and can either be **sig** for a logistic sigmoid function, **tanh** for hyperbolic tangent or **relu** for rectified linear unit.
- **INIT_RANGE**: All of the weights and biases of the neural network will be initialized uniformly in the range [**-INIT_RANGE**, **INIT_RANGE**).
- **C**: The number of classes for classification or the dimension of the output vector if regression.
- **MINIBATCH_SIZE**: The number of data points to be included in each mini-batch. Set this value to zero for full batch training.
- **NUM_HIDDEN_LAYERS**: The number of hidden layers in your neural network. Set this value to zero for a linear model (no hidden layers).
