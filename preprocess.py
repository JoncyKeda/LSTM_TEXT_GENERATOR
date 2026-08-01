import os
import re
import string
import pickle
import numpy as np
from typing import Tuple
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

DEFAULT_DATASET_PATH = "dataset/shakespeare.txt"
DEFAULT_TOKENIZER_PATH = "models/tokenizer.pickle"
DEFAULT_SEQUENCE_LENGTH = 40

def load_data(file_path: str = DEFAULT_DATASET_PATH) -> str:
    """Loads the text dataset from the specified file path.

    Args:
        file_path: Path to the text file.

    Returns:
        The raw text content of the file.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}. Please place it in the correct directory.")
        
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def preprocess_text(text: str) -> str:
    """Preprocesses raw text by converting to lowercase, removing extra spaces,
    and stripping punctuation while preserving apostrophes/contractions.

    Args:
        text: Raw text corpus.

    Returns:
        Cleaned and normalized text corpus.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    # string.punctuation is: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    punctuation_to_remove = string.punctuation.replace("'", "")
    # Remove standard punctuation characters
    clean_text = text.translate(str.maketrans("", "", punctuation_to_remove))
    
    # Normalize unicode/curly apostrophes to standard straight ones
    clean_text = clean_text.replace("’", "'").replace("`", "'")
    
    # Replace multiple spaces/newlines with a single space
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    
    return clean_text

def save_tokenizer(tokenizer: Tokenizer, file_path: str) -> None:
    """Serializes and saves a fitted Tokenizer instance to disk.

    Args:
        tokenizer: The fitted Tokenizer instance.
        file_path: Path where the tokenizer pickle file should be saved.
    """
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer automatically saved to: {file_path}")

def tokenize_and_sequence(
    text: str, 
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray, Tokenizer, int]:
    """Tokenizes clean text, partitions it into input-output sequences, and one-hot encodes targets.

    Args:
        text: Cleaned text corpus.
        sequence_length: Length of input sequences.

    Returns:
        A tuple containing:
            - X: Input sequences array of shape (num_samples, sequence_length).
            - y: One-hot encoded target outputs array of shape (num_samples, total_words).
            - tokenizer: Fitted Tokenizer instance.
            - total_words: Size of vocabulary (unique words + 1 for padding).
    """
    # Initialize and fit the tokenizer
    # By default, filters in Tokenizer includes '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    # We want to preserve single quotes/apostrophes, which are not in the default filters list anyway.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    
    # Convert text to a list of word indices
    word_sequence = tokenizer.texts_to_sequences([text])[0]
    
    # Total unique words + 1 (for zero padding index)
    total_words = len(tokenizer.word_index) + 1
    
    X = []
    y = []
    
    # Create overlapping input-output sequences
    for i in range(len(word_sequence) - sequence_length):
        X.append(word_sequence[i : i + sequence_length])
        y.append(word_sequence[i + sequence_length])
        
    # Convert to NumPy arrays
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # One-hot encode the target outputs (y)
    y_encoded = to_categorical(y_arr, num_classes=total_words)
    
    return X_arr, y_encoded, tokenizer, total_words

def preprocess_dataset(
    file_path: str = DEFAULT_DATASET_PATH,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
) -> Tuple[np.ndarray, np.ndarray, Tokenizer, int]:
    """Orchestrates the complete dataset loading, preprocessing, sequencing, and tokenizer saving pipeline.

    Args:
        file_path: Filepath to the dataset text file.
        sequence_length: Length of training input sequences.
        tokenizer_path: Filepath to save the fitted tokenizer.

    Returns:
        A tuple containing:
            - X: Preprocessed input sequences.
            - y: One-hot encoded targets.
            - tokenizer: Fitted Tokenizer.
            - total_words: Total vocabulary size.
    """
    # 1. Load raw text
    raw_text = load_data(file_path)
    
    # 2. Clean and normalize text
    clean_text = preprocess_text(raw_text)
    
    # 3. Tokenize, build sequences, and one-hot encode y
    X, y, tokenizer, total_words = tokenize_and_sequence(clean_text, sequence_length)
    
    # 4. Save tokenizer automatically
    save_tokenizer(tokenizer, tokenizer_path)
    
    return X, y, tokenizer, total_words
