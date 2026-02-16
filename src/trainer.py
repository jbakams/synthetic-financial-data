import pandas as pd
import argparse
import torch
import warnings
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def get_device_config(device_name):
    """
    Determines the correct 'cuda' parameter for SDV based on user input
    and hardware availability.
    """
    if device_name == 'cpu':
        print(" -> forcing CPU usage.")
        return False  # sdv expects cuda=False for CPU
    
    if device_name.startswith('cuda') or device_name == 'gpu':
        if torch.cuda.is_available():
            print(f" -> using GPU ({torch.cuda.get_device_name(0)})")
            return True # sdv expects cuda=True (or a string like "cuda:0")
        else:
            warnings.warn("GPU was requested, but Torch cannot find a CUDA device. Falling back to CPU.")
            return False

    return True # Default to auto-detect (True)

def train_model(device_arg='auto'):
    print(f"--- Starting Training Process [Device Preference: {device_arg}] ---")
    
    # 1. Configuration
    cuda_setting = get_device_config(device_arg)
    
    # 2. Load Data
    data_path = 'data/raw/sensitive_financial_data.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Did you run data_generator.py?")
        return

    # 3. Preprocessing: Drop PII
    # We explicitly remove the columns we NEVER want the model to learn.
    # The model will learn the correlation between Income, CreditScore, and Default.
    print("Loading data and dropping PII columns (Name, SSN, Email, Address)...")
    training_data = df.drop(columns=['Name', 'SSN', 'Email', 'Address'])

    # 4. Detect Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(training_data)
    
    print(f"Data Shape: {training_data.shape}")
    print("Training CTGAN model... (This may take time depending on your hardware)")

    # 5. Initialize and Train
    # We pass the 'cuda' parameter here to control the device
    synthesizer = CTGANSynthesizer(
        metadata, 
        cuda=cuda_setting,
        epochs=300,       # Lowered slightly for faster testing; increase to 500+ for production
        batch_size=500,
        verbose=True
    )
    
    synthesizer.fit(training_data)

    # 6. Save
    save_path = 'models/financial_synthesizer.pkl'
    synthesizer.save(save_path)
    print(f"\nSuccess! Model saved to {save_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a CTGAN model on financial data.')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['cpu', 'gpu', 'cuda', 'auto'],
                        help='Choose to train on "cpu" or "gpu" (cuda). Default is auto-detect.')

    args = parser.parse_args()
    train_model(args.device)
