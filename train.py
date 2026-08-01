import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Import modules from current project
from preprocess import preprocess_dataset, DEFAULT_SEQUENCE_LENGTH, DEFAULT_DATASET_PATH, DEFAULT_TOKENIZER_PATH
from model import build_model

DEFAULT_MODEL_SAVE_PATH = "models/best_model.keras"
DEFAULT_HISTORY_SAVE_PATH = "output/history.json"
DEFAULT_LOSS_PLOT_PATH = "output/loss.png"
DEFAULT_ACCURACY_PLOT_PATH = "output/accuracy.png"

def run_training(
    dataset_path: str = DEFAULT_DATASET_PATH,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    model_save_path: str = DEFAULT_MODEL_SAVE_PATH,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    epochs: int = 30,
    batch_size: int = 128
) -> Dict[str, Any]:
    """Orchestrates the training pipeline for the LSTM Text Generator model.

    Loads and preprocesses the corpus, splits it into training and validation sets,
    builds the network, trains it with custom callbacks (EarlyStopping, Checkpoint, ReduceLR),
    saves the training history, plots loss and accuracy curves, and prints performance summaries.

    Args:
        dataset_path: Path to the text file.
        sequence_length: Sequence window size for training inputs.
        model_save_path: Path where the best performing model should be saved.
        tokenizer_path: Path where the tokenizer pickle file should be saved.
        epochs: Number of epochs to train for.
        batch_size: Size of training batches.

    Returns:
        The training history dictionary containing metric lists.
    """
    # Ensure necessary save directories exist upfront
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # 1. Load dataset, preprocess, and fit tokenizer (saves tokenizer automatically)
    print("Loading and preprocessing dataset...")
    X, y, tokenizer, total_words = preprocess_dataset(
        file_path=dataset_path, 
        sequence_length=sequence_length, 
        tokenizer_path=tokenizer_path
    )
    
    # 2. Split dataset into train and validation sets (80% train, 20% validation)
    print("Splitting dataset into train and validation sets...")
    
    # Generate shuffled indices to ensure diverse distribution in training
    indices = np.arange(X.shape[0])
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    split_index = int(0.8 * len(X_shuffled))
    X_train, X_val = X_shuffled[:split_index], X_shuffled[split_index:]
    y_train, y_val = y_shuffled[:split_index], y_shuffled[split_index:]
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # 3. Build and compile the deep LSTM network
    print("Building and compiling deep LSTM model...")
    model = build_model(total_words, sequence_length)
    
    # 4. Configure callbacks
    # EarlyStopping prevents overfitting by halting when validation loss stops improving
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # ModelCheckpoint saves only the highest performing model weights dynamically
    model_checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    # ReduceLROnPlateau reduces the learning rate when validation loss plateaus to fine-tune learning
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )
    
    # 5. Train the model
    print("Beginning model training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    # 6. Save the training history dict
    history_data = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open(DEFAULT_HISTORY_SAVE_PATH, "w") as handle:
        json.dump(history_data, handle, indent=4)
    print(f"Training history saved to: {DEFAULT_HISTORY_SAVE_PATH}")
    
    # 7. Plot and save training charts
    print("Generating training metric curves...")
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_data["loss"], label="Training Loss", linewidth=2)
    plt.plot(history_data["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("LSTM Model Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(DEFAULT_LOSS_PLOT_PATH)
    plt.close()
    print(f"Loss plot saved to: {DEFAULT_LOSS_PLOT_PATH}")
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_data["accuracy"], label="Training Accuracy", linewidth=2)
    plt.plot(history_data["val_accuracy"], label="Validation Accuracy", linewidth=2)
    plt.title("LSTM Model Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(DEFAULT_ACCURACY_PLOT_PATH)
    plt.close()
    print(f"Accuracy plot saved to: {DEFAULT_ACCURACY_PLOT_PATH}")
    
    # 8. Print final training and validation summary
    final_loss = history_data["loss"][-1]
    final_acc = history_data["accuracy"][-1]
    val_loss = history_data["val_loss"][-1]
    val_acc = history_data["val_accuracy"][-1]
    
    best_epoch_idx = np.argmin(history_data["val_loss"])
    best_loss = history_data["loss"][best_epoch_idx]
    best_acc = history_data["accuracy"][best_epoch_idx]
    best_val_loss = history_data["val_loss"][best_epoch_idx]
    best_val_acc = history_data["val_accuracy"][best_epoch_idx]
    
    print("\n" + "="*45)
    print("TRAINING METRICS SUMMARY")
    print("="*45)
    print(f"Final Epoch Training Accuracy:   {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"Final Epoch Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Final Epoch Training Loss:       {final_loss:.4f}")
    print(f"Final Epoch Validation Loss:     {val_loss:.4f}")
    print("-"*45)
    print(f"Best Epoch (based on val_loss) - Epoch {best_epoch_idx + 1}:")
    print(f"  Training Accuracy:             {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"  Validation Accuracy:           {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Training Loss:                 {best_loss:.4f}")
    print(f"  Validation Loss:               {best_val_loss:.4f}")
    print("="*45 + "\n")
    
    return history_data

if __name__ == "__main__":
    run_training()
