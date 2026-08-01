from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def build_model(total_words: int, sequence_length: int) -> Sequential:
    """Builds and compiles a deep Sequential LSTM text generation model.

    The model maps sequence indices to dense vectors, runs them through
    a multi-layer LSTM architecture, and outputs probabilities over the vocabulary.

    Architecture:
        Input(shape=(sequence_length,))
        ↓
        Embedding(128)
        ↓
        LSTM(128, return_sequences=True)
        ↓
        Dropout(0.2)
        ↓
        LSTM(128)
        ↓
        Dense(256, activation="relu")
        ↓
        Dropout(0.2)
        ↓
        Dense(total_words, activation="softmax")

    Args:
        total_words: Size of the word vocabulary (number of unique words + 1).
        sequence_length: Number of input words in each sequence.

    Returns:
        The compiled Sequential Keras model.
    """
    model = Sequential([
        # Specify input layer explicitly to compile and build the model immediately
        Input(shape=(sequence_length,)),
        
        # Dense vector embedding
        Embedding(input_dim=total_words, output_dim=128),
        
        # Deeper LSTM recurrent structure
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        
        # Dense projection layers
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(total_words, activation="softmax")
    ])
    
    # Compile with categorical_crossentropy for one-hot encoded targets
    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Print the model summary showing output shapes and parameter counts
    model.summary()
    
    return model
