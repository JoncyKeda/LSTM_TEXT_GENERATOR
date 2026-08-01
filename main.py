import sys
import os
import pickle
from tensorflow.keras.models import load_model

from train import run_training
from generate import generate_text, DEFAULT_MODEL_PATH, DEFAULT_TOKENIZER_PATH

def display_menu() -> None:
    """Prints the primary user command-line menu interface."""
    print("===================================")
    print("LSTM TEXT GENERATOR")
    print("===================================")
    print("1. Train Model")
    print("2. Generate Text")
    print("3. Model Information")
    print("4. Exit")
    print("===================================")

def display_model_info() -> None:
    """Loads the model and tokenizer to print details and summary layout."""
    if not os.path.exists(DEFAULT_MODEL_PATH) or not os.path.exists(DEFAULT_TOKENIZER_PATH):
        print("\n[Status] Model has not been trained yet. Please run Option 1 ('Train Model') first.\n")
        return
        
    print("\nLoading model structure from disk...")
    try:
        model = load_model(DEFAULT_MODEL_PATH)
        with open(DEFAULT_TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
            
        print("\n===================================")
        print("MODEL STATUS & METADATA")
        print("===================================")
        print(f"Model File:         {DEFAULT_MODEL_PATH}")
        print(f"Tokenizer File:     {DEFAULT_TOKENIZER_PATH}")
        print(f"Vocabulary Size:    {len(tokenizer.word_index) + 1} words")
        print(f"Input Sequence Len: {model.input_shape[1]} words")
        
        model_size_mb = os.path.getsize(DEFAULT_MODEL_PATH) / (1024 * 1024)
        tokenizer_size_kb = os.path.getsize(DEFAULT_TOKENIZER_PATH) / 1024
        print(f"Model File Size:    {model_size_mb:.2f} MB")
        print(f"Tokenizer File Size: {tokenizer_size_kb:.2f} KB")
        print("-----------------------------------")
        print("Model Architecture Summary:")
        model.summary()
        print("===================================\n")
    except Exception as e:
        print(f"\n[Error] Failed to load model details: {e}\n")

def main() -> None:
    """Primary command-line interface entry point loop."""
    while True:
        display_menu()
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\nInitializing model training process...")
            try:
                run_training()
                print("\n[Success] Training complete! Metrics and charts generated in output/\n")
            except Exception as e:
                print(f"\n[Error] An error occurred during training: {e}\n")
                
        elif choice == "2":
            seed_text = input("\nEnter the starting seed phrase: ").strip()
            if not seed_text:
                print("[Input Error] Seed phrase cannot be empty.\n")
                continue
                
            try:
                next_words = int(input("Enter number of words to generate: "))
                if next_words <= 0:
                    print("[Input Error] Number of words must be a positive integer.\n")
                    continue
            except ValueError:
                print("[Input Error] Invalid numeric entry for word count.\n")
                continue
                
            try:
                temp_input = input("Enter temperature (default 0.8, range 0.1-2.0): ").strip()
                temperature = 0.8
                if temp_input:
                    temperature = float(temp_input)
                    if temperature <= 0.0:
                        print("[Input Error] Temperature must be greater than zero. Using default (0.8).\n")
                        temperature = 0.8
            except ValueError:
                print("[Input Error] Invalid temperature format. Using default (0.8).\n")
                temperature = 0.8
                
            print("\nPredicting and generating text...")
            try:
                result = generate_text(
                    seed_text=seed_text, 
                    next_words=next_words, 
                    temperature=temperature
                )
                print("\n--- Generated Text ---")
                print(result)
                print("----------------------\n")
            except FileNotFoundError as e:
                print(f"\n[Missing Assets] {e}")
                print("Please run Option 1 ('Train Model') first.\n")
            except Exception as e:
                print(f"\n[Error] An error occurred during text generation: {e}\n")
                
        elif choice == "3":
            display_model_info()
            
        elif choice == "4":
            print("\nExiting. Goodbye!")
            sys.exit(0)
            
        else:
            print(f"\n[Input Error] '{choice}' is not a valid selection. Please choose 1, 2, 3, or 4.\n")

if __name__ == "__main__":
    main()
