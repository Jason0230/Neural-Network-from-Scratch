import numpy as np
import numpy.typing as npt
import keras
from abc import ABC, abstractmethod
from utils import apply_activation, activation_prime, softmax_prime

# Abstract Layer class

class Layer(ABC):

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, delta_next: np.ndarray) -> np.ndarray:
        pass


class Dense(Layer):

    def __init__(self, units: int, activation: str):
        self.output_units = units
        self.input_units = -1
        self.activation = activation

    # initializes weight and bias matrices based on input and output sizes, values are randomized from scaler times normal distribution
    def init_matrices(self):
        """
        Initializes the weight and bias matrices for the Dense layer.
        Weights are initialized using He initialization.
        Biases are initialized to zero.
        
        Weight matrix shape: (output_units, input_units).
        Bias matrix shape: (output_units, 1).
        """

        # weights is (output size x input size)
        self.W = np.random.randn(self.output_units, self.input_units) * np.sqrt(2. / self.input_units)

        # bias matrix is (output size x 1)
        self.B = np.zeros((self.output_units, 1))

        # initalize dW and db to 0
        self.dW = np.zeros((self.output_units, self.input_units))
        self.db = np.zeros((self.output_units, 1))
    
    # Forward propgation given an input activation, returns the activation vector for next layer
    # input is (input_size x 1) vector
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation through the Dense layer.
        Saves the input activation (a^l-1), pre-activation (z), and output activation (a) for use in backpropagation.
        
        Args:
            X (np.ndarray): Input activation from previous layer of shape (input_units, 1).

        Returns:
            np.ndarray: Output activation for next layer of shape (output_units, 1).
        """

        # saves values for backprop
        self.prev_a = X
        self.z = self.W @ X + self.B  # z = a^(l-1)W + b
        self.a = apply_activation(self.activation, self.z)

        return self.a
    
    # Performs back propgation and returns the backpropagation error (dC/dz)
    # the parameter is the backpropation error (dC/d(z+1)) from the next layer
    # Assumes the layer is NOT the output layer
    def backward(self, delta_next: np.ndarray) -> np.ndarray:
        """
        Performs backward propagation through the Dense layer.
        Computes gradients and returns the error for the current layer for the previous layer.

        Args:
            delta_next (np.ndarray): Error from the next layer (dC/dz^(l+1)).

        Returns: 
            np.ndarray: Error for this layer (dC/dz^(l)).
        """
        # delta_next = dC/dz^(l)

        if self.activation == 'softmax':
            # special case for softmax since its derivative is a jacobian matrix

            # Jacobian matrix for softmax
            jacobian = softmax_prime(self.a)

            delta = jacobian @ delta_next  # dC/dz^(l)
        else:
            # dC/dz^(l)
            delta = (delta_next) * activation_prime(self.activation, self.z).reshape(-1, 1)

        # add to dW and dB for gradient descent

        self.dW += delta @ self.prev_a.T
        self.db += delta

        delta = self.W.T @ delta  # dC/dz^(l)

        return delta
    
    # beta is for momentum optimizer
    # beta2 and epsilon are for rms and adam optimizer
    def update_weights(self, batch_size: int, optimizer: str, lr: float = .001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7) -> None:
        """
        Performs gradient descent updates the weights and biases of the Dense layer using the specified optimizer.
        Resets the accumulated gradients after the update.
        
        Args:
            batch_size (int): The size of the batch used to compute the gradients.
            optimizer (str): The optimization algorithm to use ('sgd', 'momentum', 'rmsprop', or 'adam').
            lr (float): Learning rate. Default is 0.001.
            beta1 (float): Beta 1 for momentum and Adam. Default is 0.9.
            beta2 (float): Beta 2 for RMSprop and Adam. Default is 0.999.
            epsilon (float): Epsilon for RMSprop and Adam. Default is 1e-7.
        """
        if batch_size == None:
            batch_size = 1

        # calculate average dW and db over batch
        self.dW /= batch_size
        self.db /= batch_size

        match optimizer:
            case 'momentum':
                if not hasattr(self, 'v_dW'):
                    self.v_dW = np.zeros_like(self.dW)
                    self.v_db = np.zeros_like(self.db)

                self.v_dW = beta1 * self.v_dW + (1 - beta1) * (self.dW)
                self.v_db = beta1 * self.v_db + (1 - beta1) * (self.db)

                self.W -= lr * self.v_dW
                self.B -= lr * self.v_db

            case 'rmsprop':
                if not hasattr(self, 's_dW'):
                    self.s_dW = np.zeros_like(self.dW)
                    self.s_db = np.zeros_like(self.db)

                self.s_dW = beta2 * self.s_dW + (1 - beta2) * np.square(self.dW)
                self.s_db = beta2 * self.s_db + (1 - beta2) * np.square(self.db)

                self.W -= lr * (self.dW) / (np.sqrt(self.s_dW) + epsilon)
                self.B -= lr * (self.db) / (np.sqrt(self.s_db) + epsilon)

            case 'adam':
                if not hasattr(self, 'v_dW'):
                    self.v_dW = np.zeros_like(self.dW)
                    self.v_db = np.zeros_like(self.db)
                    self.s_dW = np.zeros_like(self.dW)
                    self.s_db = np.zeros_like(self.db)

                # momentum step
                self.v_dW = beta1 * self.v_dW + (1 - beta1) * (self.dW)
                self.v_db = beta1 * self.v_db + (1 - beta1) * (self.db)

                # rmsprop step
                self.s_dW = beta2 * self.s_dW + (1 - beta2) * np.square(self.dW)
                self.s_db = beta2 * self.s_db + (1 - beta2) * np.square(self.db)

                # bias correction
                v_dW_corrected = self.v_dW / (1 - beta1)
                v_db_corrected = self.v_db / (1 - beta1)
                s_dW_corrected = self.s_dW / (1 - beta2)
                s_db_corrected = self.s_db / (1 - beta2)

                self.W -= lr * (v_dW_corrected) / (np.sqrt(s_dW_corrected) + epsilon)
                self.B -= lr * (v_db_corrected) / (np.sqrt(s_db_corrected) + epsilon)

            case _:
                self.W -= lr * (self.dW)
                self.B -= lr * (self.db)

        # reset dW and dB
        self.dW.fill(0.0)
        self.db.fill(0.0)

class Input(Layer):

    def __init__(self, units):
        self.units = units
        self.input = None

    # Input layer is just input so return input for next layer
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X

    def backward(self, delta_next: np.ndarray) -> np.ndarray:
        return delta_next
    
    def update_weights(self, batch_size, optimizer, lr = .001, beta = 0.9):
        pass
    
