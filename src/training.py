import time
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix
from typing import Tuple, Dict, Any, List

class ConfusionMatrixSaver(tf.keras.callbacks.Callback):
    """Custom callback to compute and save the confusion matrix after each epoch."""
    
    def __init__(self, validation_data: Tuple[np.ndarray, np.ndarray], filepath: Path):
        super().__init__()
        self.x_val, self.y_val_true = validation_data
        self.filepath = filepath
        self.confusion_matrices = []

    def on_epoch_end(self, epoch: int, logs: dict = None):
        # Predict probabilities and convert to class indices
        y_pred_probs = self.model.predict(self.x_val, verbose=0)
        
        y_true = np.argmax(self.y_val_true, axis=1) if self.y_val_true.ndim > 1 else self.y_val_true
        y_pred = np.argmax(y_pred_probs, axis=1) if y_pred_probs.ndim > 1 else (y_pred_probs > 0.5).astype(int)

        # Calculate and store
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices.append(cm)

        # Save to disk
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.confusion_matrices, f)


def calculate_convergence_epoch(val_accuracy: List[float], window_size: int = 10) -> int:
    """Estimates the convergence epoch using a smoothed moving average."""
    val_acc = np.array(val_accuracy)
    epochs = len(val_acc)
    
    if epochs < window_size:
        return epochs

    # Moving average
    rolling_mean = np.convolve(val_acc, np.ones(window_size) / window_size, mode='valid')
    threshold = 0.99 * np.max(val_acc)
    converged_indices = np.where(rolling_mean > threshold)[0]
    
    if len(converged_indices) > 0:
        return int(converged_indices[0] + window_size - 1)
    
    return epochs


def train_and_evaluate(
    model: tf.keras.Model,
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    batch_size: int,
    experiment_name: str = "model_run",
    save_cm: bool = False,
    output_dir: str | Path = "RawNetworkData"
) -> Dict[str, Any]:
    """
    Trains the model, measures execution times, evaluates performance, 
    and returns a comprehensive results dictionary.
    """
    x_train, y_train = train_data
    x_test, y_test = test_data
    out_path = Path(output_dir)
    
    # Setup callbacks
    callbacks = []
    if save_cm:
        cm_file = out_path / "ConfusionMatrix" / f"{experiment_name}.pkl"
        callbacks.append(ConfusionMatrixSaver(validation_data=test_data, filepath=cm_file))

    # --- 1. Training Phase ---
    t0 = time.time()
    history = model.fit(
        x=x_train, y=y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )
    train_time = time.time() - t0

    # --- 2. Evaluation Phase ---
    t0 = time.time()
    predictions = model.predict(x_test, verbose=0)
    eval_time = time.time() - t0

    # Metrics computation
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.mean(y_pred == y_true)
    
    # Extract validation accuracy to compute convergence
    val_acc_history = history.history.get('val_accuracy', [])
    convergence_epoch = calculate_convergence_epoch(val_acc_history)

    print(f"[{experiment_name}] Train Time: {train_time:.2f}s | Eval Time: {eval_time:.2f}s | Acc: {accuracy:.4f} | Converged @ Epoch: {convergence_epoch}")

    # Return structured results
    return {
        "history": history.history,
        "train_time": train_time,
        "eval_time": eval_time,
        "predictions": predictions,
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "convergence_epoch": convergence_epoch
    }


if __name__ == "__main__":
    # Example usage for testing the module
    from models import build_mlp_model
    import numpy as np
    
    print("Testing training module with dummy data...")
    # Dummy data
    x_dummy = np.random.rand(100, 784).astype(np.float32)
    y_dummy = np.eye(47)[np.random.choice(47, 100)]
    
    # Build model
    dummy_model = build_mlp_model(
        input_shape=(784,), output_dim=47, hidden_layer_sizes=[128]
    )
    
    # Train
    results = train_and_evaluate(
        model=dummy_model,
        train_data=(x_dummy, y_dummy),
        test_data=(x_dummy, y_dummy), # Using train as test just for this dummy run
        epochs=2,
        batch_size=32,
        experiment_name="dummy_test",
        save_cm=False
    )
    
    print(f"Test run completed. Final Accuracy: {results['accuracy']}")