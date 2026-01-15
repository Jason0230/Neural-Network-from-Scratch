from typing import Optional
import Layers
from utils import activation_prime, get_loss, get_loss_gradient, train_val_split
import numpy as np

# Class that represents the Neural Network model, contains layers and methods for training
# First Layer for this network must be an Input Layer
# if a Dense layer is given as the first layer it will be converted to an Input Layer
class Network:

    def __init__(self, layers: list[Layers.Layer] | tuple[Layers.Layer,...] | None = None):
        """
        Initializes the Network with optional layers.
        Layers can be added later using the add_layer method.
        Layers should be instances of Layer subclasses (e.g., Input, Dense). And the first layer must be an Input layer.
        If a Dense layer is provided as the first layer, it will be converted to an Input layer.
        Input Layers cannot be used in hidden layers or output layers.
        
        Args:
            layers (list[Layers.Layer] | tuple[Layers.Layer,...] | None): List or Tuple of layers to initialize the network with. Default is None.
        """
        self.layers = []

        if layers is not None:
            for layer in layers:
                self.add_layer(layer)

    def add_layer(self, layer: Layers.Layer) -> None:
        """
        Adds a layer to the network and updates parameters accordingly.
        If the Network is empty, and the added layer is a Dense layer, it will be converted to be an Input layer.
        
        Args:
            layer (Layers.Layer): The layer to be added to the network.
        """
        
        # Check if Input Layer
        if self.layers == []:
            
            if type(layer) == Layers.Input:
                self.layers.append(layer)
            # Convert Dense into Input
            elif type(layer) == Layers.Dense:
                self.layers.append(self._convert_Dense_to_Input(layer))
        
        # hidden layer with dense
        elif type(layer) == Layers.Dense:
            self.layers.append(layer)
        
        # Input layer not first layer, throw exception
        else:
            raise Exception("Input Layer cannot be a Hidden Layer!")
        
        # update parameters after adding new layer such that the dimensions are correct for the weight and bias matrices
        self.update_params()
        
    def update_params(self) -> None:
        """
        Updates the parameters of all layers in the network based on the current layer units.
        This function ensures that each layer's input size matches the output size of the previous layer.
        
        E.g updates the the input units of layer l to be equal to the output units of layer l-1.
        """
        if self.layers != []:
            prev = self.layers[0].units

            for layer in self.layers[1:]:
                layer.input_units = prev

                # update the weight and bias matrices based on new input
                layer.init_matrices()

                prev = layer.output_units

    def compile(self, loss_func: str, learning_rate: float = 0.001, optimizer: Optional[str] = None) -> None:
        """
        Compiles the model with the specified loss function, learning rate, and optimizer.
        Required before training the model.
        
        Args:
            loss_func (str): The loss function to use.
            learning_rate (float): The learning rate for the optimizer. Default is 0.001.
            optimizer (Optional[str]): The optimization algorithm to use. Default is None.
        """
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: Optional[int] = None, epochs: int = 1, validation_set: Optional[tuple[np.ndarray, np.ndarray]] = None, validation_split: Optional[float] = None) -> dict:
        """
        Trains the model using the provided training data and labels.
        Requires the model to be compiled first using the compile method.

        Args:
            x (np.ndarray): Training input data.
            y (np.ndarray): Training target data.
            batch_size (Optional[int]): Number of training samples to process before updating weights. Default is None (process all samples before updating weights).
            epochs (int): Number of epochs to train the model. Default is 1.
            validation_set (Optional[tuple[np.ndarray, np.ndarray]]): A tuple containing validation input data and labels, Overrides validation split. Default is None.
            validation_split (Optional[float]): Proportion of the training data to use for validation (between 0 and 1). Default is None (no validation set).

        Returns:
            dict: A dictionary containing loss history for each epoch. If validation_split is provided, includes validation loss history as well. 
        """

        # check if compiled
        if self.loss_func == None:
            raise Exception("Model was not compiled")
        
        num_x, input_size = x.shape
        num_y, output_size = y.shape

        n = len(self.layers)

        # Check if each input has a label
        if num_x != num_y:
            raise Exception("input and outputs don't have same amount of data")
        
        # Check if the dimensions of the data match the dimensions of the network
        elif self.layers[0].units != input_size or self.layers[n-1].output_units != output_size:
            raise Exception("input size or output size are not the same as the neural network")
        
        if validation_set is not None:
            x_val, y_val = validation_set
            val_num = x_val.shape[0]

        elif validation_split is not None:
            x, y, x_val, y_val = train_val_split(x, y, validation_split)
            num_x = x.shape[0]
            val_num = x_val.shape[0] if validation_split is not None else 0

        train_num: int = num_x

        loss_history = []
        val_loss_history = []
        
        # Training for given number of epochs
        for curr_epoch in range(epochs):

            # the accumalated loss for all batch loss for this epoch
            running_loss = 0.0

            # Processing every training data
            for i in range(train_num): 

                #* Forward Propagation
                activation = (x[i]).reshape(-1,1)

                for layer in self.layers:
                    activation = layer.forward(activation)

                output = activation

                #* Back Propagation

                delta = get_loss_gradient(self.loss_func, output, (y[i]).reshape(-1,1))
                running_loss += get_loss(self.loss_func, output, (y[i]).reshape(-1,1))

                # propagate error backwards
                for layer in reversed(self.layers):
                    delta = layer.backward(delta)

                # update weights and bias matrices after processing batch_size number of data
                if batch_size is not None and i % batch_size == 0:
                    for layer in self.layers:
                        layer.update_weights(batch_size, self.optimizer, lr= self.learning_rate)
        
            # update weights if batch_size is None or number of data is not divisible by batch_size
            for layer in self.layers:
                layer.update_weights(batch_size, self.optimizer, lr= self.learning_rate)
            
            epoch_loss: float = running_loss / train_num
            
            # Calculate loss for validation data if validation split available or validation set provided
            if validation_split is not None or validation_set is not None:
                val_loss = 0
                for i in range(val_num):
                    activation = (x_val[i]).reshape(-1,1)

                    for layer in self.layers:
                        activation = layer.forward(activation)

                    output = activation
                    val_loss += get_loss(self.loss_func, output, (y_val[i]).reshape(-1,1))
                
                val_loss /= val_num

                print(f"Epoch {curr_epoch}, Loss: {epoch_loss:.5f}, Val Loss: {val_loss:.5f}")
                loss_history.append(epoch_loss)
                val_loss_history.append(val_loss)

            else:
                # Print loss at end of epoch
                print(f"Epoch {curr_epoch}, Loss: {epoch_loss:.5f}")
                loss_history.append(epoch_loss)
        
        return {"loss": loss_history, "val_loss": val_loss_history} if validation_split is not None or validation_set is not None else {"loss": loss_history}
            

    # Makes a prediction given input data list
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions for the given input data by performing a forward pass through the network.
        Resulting predictions are returned as a NumPy array.

        Args:
            x (np.ndarray): Input data for which to make predictions. With shape (num_samples, num_features).

        Returns:
            np.ndarray: Predicted output data with shape (num_samples, output_units).
        """
        
        x = np.array(x)
        n_samples = x.shape[0]

        predictions = []

        # Process data in chunks
        for i in range(0, n_samples):
            
            # Forward pass through all layers
            activation = x[i].reshape(-1, 1)
            for layer in self.layers:
                activation = layer.forward(activation)
            
            predictions.append(activation.reshape(-1))

        # Stack the batch results back into a single array
        return np.array(predictions)

    def _convert_Dense_to_Input(self, layer: Layers.Dense) -> Layers.Input:
        """
        Converts a Dense layer into an Input layer based on its output units.
        
        Args:
            layer (Layers.Dense): The Dense layer to be converted.
        
        Returns:
            Layers.Input: The converted Input layer.
        """
        return Layers.Input(units= layer.output_units)
    
    def to_string(self):
        for layer in self.layers:
            if type(layer) == Layers.Input:
                print(f"Input Layer, {layer.units} inputs")
            elif type(layer) == Layers.Dense:
                print(f"Dense Layer, Input: {layer.input_units}, Output: {layer.output_units}")
