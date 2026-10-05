import numpy as np
from pathlib import Path
from typing import Tuple, List

def get_dataset_info(dataset_name: str) -> Tuple[int, List[str]]:
    """Returns the number of classes and the symbol mapping for the selected EMNIST dataset."""
    if dataset_name == "balanced":
        symbols = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
        ]
        return 47, symbols
        
    elif dataset_name == "digits":
        symbols = [str(i) for i in range(10)]
        return 10, symbols
        
    elif dataset_name == "letters":
        symbols = ['_'] + [chr(i) for i in range(97, 123)]
        return 27, symbols
        
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. Allowed: 'balanced', 'digits', 'letters'.")


def load_emnist_data(
    data_dir: str | Path, 
    dataset_name: str = "balanced", 
    normalize: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], List[str]]:
    """Loads, reshapes, and normalizes the data, and applies one-hot encoding to the labels."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Load raw data
    try:
        x_train = np.load(data_path / f"emnist-{dataset_name}-train-images.npy")
        y_train = np.load(data_path / f"emnist-{dataset_name}-train-labels.npy")
        x_test  = np.load(data_path / f"emnist-{dataset_name}-test-images.npy")
        y_test  = np.load(data_path / f"emnist-{dataset_name}-test-labels.npy")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Files for '{dataset_name}' not found. Error: {e}")

    # Flattening (2D to 1D vector)
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    # Scale to [0, 1]
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    # One-hot encode labels
    num_classes, symbol_mapping = get_dataset_info(dataset_name)
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]

    return (x_train, y_train_onehot), (x_test, y_test_onehot), symbol_mapping


if __name__ == "__main__":
    # Quick functional test
    test_dir = "/usr/local/share/teach/ML/EMNIST/numpy/"
    (X_tr, Y_tr), (X_te, Y_te), syms = load_emnist_data(test_dir, "balanced")
    print(f"X_train: {X_tr.shape} | Y_train: {Y_tr.shape} | Classes: {len(syms)}")