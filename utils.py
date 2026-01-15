import numpy as np

def relu(vec: np.ndarray) -> np.ndarray:
    """
    Applies the ReLU activation function on each element of the input vec.
    
    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Output vector after applying ReLU.
    """
    return np.maximum(0, vec)

def relu_prime(vec: np.ndarray) -> np.ndarray:
    """
    Applies the derivative of the ReLU activation function on each element of the input vec.
    Since the derivative of ReLU is a diagonal jacobian matrix, this function returns a vector where each element
    corresponds to the derivative of ReLU at that position for element-wise multiplication.

    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Output vector after applying the derivative of ReLU.
    """
    return (vec > 0).astype(float)

# applies softmax on a vector
def softmax(vec: np.ndarray) -> np.ndarray:
    """
    Applies the softmax activation function on the input vector.
    
    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Output vector after applying softmax.
    """

    # converts every element to e^x
    vec = np.exp(vec - np.max(vec))  # for numerical stability

    return vec / np.sum(vec)
    
def softmax_prime(vec: np.ndarray) -> np.ndarray:
    """
    Applies the derivative of the softmax activation function on the input vector.
    Note that the derivative of softmax is a Jacobian matrix.

    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Jacobian matrix after applying the derivative of softmax.
    """
    # ensure vec is a column vector
    vec = vec.reshape(-1,1)

    return np.diagflat(vec) - (vec @ vec.T)

def sigmoid(vec: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid activation function on each element of the input vec.
    
    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Output vector after applying sigmoid.
    """
    return 1 / (1 + np.exp(-vec))

def sigmoid_prime(vec: np.ndarray) -> np.ndarray:
    """
    Applies the derivative of the sigmoid activation function on each element of the input vec.
    
    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Output vector after applying the derivative of sigmoid.
    """
    sig = sigmoid(vec)
    return sig * (1 - sig)

def tanh(vec: np.ndarray) -> np.ndarray:
    """
    Applies the tanh activation function on each element of the input vec.
    
    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Output vector after applying tanh.
    """
    return np.tanh(vec)

def tanh_prime(vec: np.ndarray) -> np.ndarray:
    """
    Applies the derivative of the tanh activation function on each element of the input vec.
    
    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Output vector after applying the derivative of tanh.
    """
    return 1 - np.tanh(vec) ** 2
    
def apply_activation(activation: str, input: np.ndarray) -> np.ndarray:
    """
    Applies the specified activation function on the input vector.
    Contains all supported activation functions.
    Currently supports 'relu', 'softmax', 'sigmoid', and 'tanh'.
    
    Args:
        activation (str): The activation function to apply.
        input (np.ndarray): Input vector.
    
    Returns:
        np.ndarray: Output vector after applying the activation function.
    """

    match str.lower(activation):
        case 'relu':
            return relu(input)
        
        case 'softmax':
            return softmax(input)
        
        case 'sigmoid':
            return sigmoid(input)
        
        case 'tanh':
            return tanh(input)
        
        case _:
            raise Exception(f"Activation function {activation} is not supported")

def activation_prime(activation: str, input: np.ndarray) -> np.ndarray:
    """
    Applies the derivative of the specified activation function on the input.
    Contains all supported activation functions.
    Currently supports 'relu', 'softmax', 'sigmoid', and 'tanh'.
    
    Args:
        activation (str): The activation function whose derivative to apply.
        input (np.ndarray): Input vector.
    
    Returns:
        np.ndarray: Output vector or matrix after applying the derivative of the activation function.
    """
    match str.lower(activation):
        case 'relu':
            return relu_prime(input)
        
        case 'softmax':
            return softmax_prime(input)
        
        case 'sigmoid':
            return sigmoid_prime(input)
        
        case 'tanh':
            return tanh_prime(input)
        
        case _:
            raise Exception(f"Activation function {activation} is not supported")

def get_loss(loss_func: str, predicted: np.ndarray, expected: np.ndarray | float) -> float:
    """
    Calculates the loss based on the specified loss function.

    Args:
        loss_func (str): The loss function to use.
        predicted (np.ndarray): Predicted values.
        expected (np.ndarray | float): Expected values.

    Returns:
        float: The calculated loss.
    """

    match str.lower(loss_func):
        case "mean_squared_error":
            return np.mean(np.square(predicted - expected))
        
        case "mean_absolute_error":
            return np.mean(np.abs(predicted - expected))

        # one hot encoded, expected should be a vector
        case "categorical_crossentropy":
            return -np.sum(expected * np.log(predicted))

        # not one hot encoded, expected should be scaler
        case "sparse_categorical_crossentropy":
            return -np.log(predicted[int(expected)])

        case _:
            raise Exception(f"Neural Network currently doesn't support {loss_func}")

def get_loss_gradient(loss_func: str, predicted: np.ndarray | float, expected: np.ndarray | float) -> np.ndarray | float:
    """
    Calculates the gradient of the loss function with respect to the predicted values.
    
    Args:
        loss_func (str): The loss function to use.
        predicted (np.ndarray | float): Predicted values.
        expected (np.ndarray | float): Expected values.

    Returns:
        np.ndarray | float: The gradient of the loss with respect to the predicted values.
    """

    match str.lower(loss_func):
        case "mean_squared_error":
            return 2 * (predicted - expected)
        case "mean_absolute_error":
            return np.sign(predicted - expected)

        # one hot encoded, expected should be a vector
        case "categorical_crossentropy":
            return -expected / predicted

        # not one hot encoded, expected should be scaler
        case "sparse_categorical_crossentropy":
            return -1 / predicted[int(expected)]
            
        case _:
            raise Exception(f"Neural Network currently doesn't support {loss_func}")

# helper function to split data into validation and training sets
def train_val_split(x: np.ndarray, y: np.ndarray, val_split: float, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training and validation sets based on the specified validation split ratio.
    Randomly shuffles the data before splitting if shuffle is True.

    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Corresponding Output data.
        val_split (float): Proportion of the data to be used for validation (between 0 and 1).
        shuffle (bool): Whether to shuffle the data before splitting. Default is True.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and validation data splits in order training_x, training_y, validation_x, validation_y.
    """

    # shuffle data before splitting
    if shuffle:
        indices = np.random.permutation(x.shape[0])
        x = x[indices]
        y = y[indices]

    num_data = x.shape[0]
    split_index = int(num_data * (1 - val_split))

    x_train = x[:split_index]
    y_train = y[:split_index]

    x_val = x[split_index:]
    y_val = y[split_index:]
    
    return x_train, y_train, x_val, y_val
        