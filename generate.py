import os
import pickle
import numpy as np
from typing import Tuple, List
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocess import DEFAULT_TOKENIZER_PATH, DEFAULT_SEQUENCE_LENGTH

DEFAULT_MODEL_PATH = "models/best_model.keras"
DEFAULT_OUTPUT_TEXT_PATH = "output/generated_text.txt"

# Internal caches to prevent re-reading files on successive calls
_model = None
_tokenizer = None

def load_assets(
    model_path: str = DEFAULT_MODEL_PATH, 
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
) -> Tuple[Sequential, Tokenizer]:
    """Loads and caches the compiled Keras model and fitted tokenizer from disk.

    Args:
        model_path: Path to the trained Keras model.
        tokenizer_path: Path to the pickled Tokenizer.

    Returns:
        A tuple of (loaded_model, loaded_tokenizer).

    Raises:
        FileNotFoundError: If either model or tokenizer is missing.
    """
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Please train the model first."
            )
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer asset not found at {tokenizer_path}. Please train the model first."
            )
            
        print("Loading trained model and tokenizer...")
        _model = load_model(model_path)
        with open(tokenizer_path, "rb") as handle:
            _tokenizer = pickle.load(handle)
            
    return _model, _tokenizer

def generate_text(
    seed_text: str,
    next_words: int,
    temperature: float = 0.8,
    repetition_penalty: float = 1.5,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    model_path: str = DEFAULT_MODEL_PATH,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    output_path: str = DEFAULT_OUTPUT_TEXT_PATH,
    min_confidence: float = 0.01  # Retained for signature compatibility, but unused
) -> str:
    """Generates text word-by-word starting from the provided seed text.

    Uses temperature-scaled sampling, ignores padding tokens, prevents
    immediate repetition of the same word, and fallback to top 5 predictions 
    if a choice is invalid. Never halts early.

    Args:
        seed_text: The initial text string to seed the generator.
        next_words: Number of words to generate.
        temperature: Temperature value for sampling. Higher values mean more random,
            lower values mean more deterministic.
        repetition_penalty: Logit penalty applied to recently generated words to prevent loops.
        sequence_length: The length of training input sequences.
        model_path: Path to the model.
        tokenizer_path: Path to the tokenizer.
        output_path: Path to write the final generated text file.
        min_confidence: Deprecated/unused. Re-added to prevent signature mismatch.

    Returns:
        The combined seed and generated text.
    """
    model, tokenizer = load_assets(model_path, tokenizer_path)
    current_text = seed_text
    
    # Store indices of recently generated words to penalize repetition
    recent_indices: List[int] = []
    
    print(f"Generating {next_words} words with temperature={temperature}...")
    
    for _ in range(next_words):
        # Convert the current text sequence to integer tokens
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        
        # Pre-pad sequence to match expected input dimension (e.g. 40 words)
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")
        
        # Predict class probabilities
        predictions = model.predict(token_list, verbose=0)
        probs = predictions[0]
        
        # Avoid log(0) issues by adding tiny epsilon
        logits = np.log(probs + 1e-10)
        
        # Ignore the padding token (index 0) by setting its logit to a very low value
        logits[0] = -1e9
        
        # Prevent immediate repetition of the same word (e.g., "the the")
        last_word_index = recent_indices[-1] if recent_indices else None
        if last_word_index is not None:
            logits[last_word_index] = -1e9
            
        # Sample index using temperature scaling
        if temperature <= 1e-5:
            # Fall back to greedy search if temperature is near zero
            predicted_index = int(np.argmax(logits))
        else:
            scaled_logits = logits / temperature
            # Shift for numerical stability in exponentiation
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            new_probs = exp_logits / np.sum(exp_logits)
            predicted_index = int(np.random.choice(len(new_probs), p=new_probs))
            
        # Lookup the word from the index mapping
        predicted_word = tokenizer.index_word.get(predicted_index, "")
        
        # If the predicted word is invalid or empty, randomly choose from the top 5 predicted words
        if not predicted_word:
            # Find top 5 predicted indices based on original probabilities
            # We exclude padding (0) and immediate repetition if possible
            valid_top_indices = []
            for idx in np.argsort(probs)[::-1]:
                if idx == 0:
                    continue
                if last_word_index is not None and idx == last_word_index:
                    continue
                word = tokenizer.index_word.get(idx, "")
                if word:
                    valid_top_indices.append(idx)
                    if len(valid_top_indices) == 5:
                        break
            
            # Select randomly from the valid top 5 predicted indices
            if valid_top_indices:
                predicted_index = int(np.random.choice(valid_top_indices))
                predicted_word = tokenizer.index_word.get(predicted_index, "")
            else:
                # Absolute fallback if no valid words can be retrieved
                predicted_index = 1
                predicted_word = tokenizer.index_word.get(predicted_index, "the")
                
        # Add to recently generated words and update output text
        recent_indices.append(predicted_index)
        current_text += " " + predicted_word
        
    # Automatically save generated text to output folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(current_text)
    print(f"Generated text automatically saved to: {output_path}")
        
    return current_text

def main() -> None:
    """Executes a test run of the text generation module."""
    seed_phrase = "shall i compare thee to a summers day"
    words_to_generate = 20
    
    print(f"Starting test generation with seed phrase: '{seed_phrase}'")
    try:
        result = generate_text(
            seed_text=seed_phrase, 
            next_words=words_to_generate,
            temperature=0.8
        )
        print("\n--- Test Generation Result ---")
        print(result)
        print("------------------------------\n")
    except FileNotFoundError as e:
        print(f"\n[Asset Error] {e}")
        print("Please train the model first by running `python train.py` to generate the checkpoint files.\n")
    except Exception as e:
        print(f"\n[Execution Error] {e}")

if __name__ == "__main__":
    main()
