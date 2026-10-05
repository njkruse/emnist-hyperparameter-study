import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD, Adam
from typing import List, Tuple, Union

def build_mlp_model(
    input_shape: Tuple[int, ...],
    output_dim: int,
    hidden_layer_sizes: List[int],
    hidden_activations: Union[str, List[str]] = "relu",
    output_activation: str = "softmax",
    learning_rate: float = 0.1,
    loss_function: str = "categorical_crossentropy",
    optimizer_type: str = "SGD"
) -> tf.keras.Model:
    """
    Builds and compiles a Multi-Layer Perceptron (MLP) for classification or regression.
    
    Args:
        input_shape: Shape of the input data (e.g., (784,) for flattened EMNIST).
        output_dim: Number of output neurons (e.g., 47 for EMNIST balanced).
        hidden_layer_sizes: List containing the number of neurons for each hidden layer.
        hidden_activations: Activation function(s) for the hidden layers. 
                            Can be a single string or a list matching hidden_layer_sizes.
        output_activation: Activation function for the output layer.
        learning_rate: Learning rate for the optimizer.
        loss_function: Loss function to use (e.g., 'mse' or 'categorical_crossentropy').
        optimizer_type: Optimizer to use ('SGD' or 'Adam').
        
    Returns:
        A compiled tf.keras.Model.
    """
    
    # Standardize hidden_activations to a list
    if isinstance(hidden_activations, str):
        hidden_activations = [hidden_activations] * len(hidden_layer_sizes)
    elif len(hidden_activations) != len(hidden_layer_sizes):
        raise ValueError("Length of hidden_activations must match length of hidden_layer_sizes.")

    # Initialize Sequential model
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Add hidden layers
    for units, activation in zip(hidden_layer_sizes, hidden_activations):
        model.add(Dense(units, activation=activation))
        
    # Add output layer
    model.add(Dense(output_dim, activation=output_activation))
    
    # Configure optimizer
    if optimizer_type.upper() == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_type.upper() == "ADAM":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}. Use 'SGD' or 'Adam'.")
        
    # Compile model
    model.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    # Quick functional test
    print("Testing model builder...")
    test_model = build_mlp_model(
        input_shape=(784,),
        output_dim=47,
        hidden_layer_sizes=[2048, 256],
        hidden_activations=["relu", "tanh"],
        output_activation="softmax",
        learning_rate=0.01,
        loss_function="categorical_crossentropy"
    )
    
    test_model.summary()
    print("Model module works perfectly!")