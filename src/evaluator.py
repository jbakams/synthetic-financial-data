import pandas as pd
import numpy as np
import argparse
import pickle
import os

# SDV Evaluation
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

# Machine Learning Utility
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_real_data():
    """Loads the original sensitive data and drops PII for comparison."""
    df = pd.read_csv('data/raw/sensitive_financial_data.csv')
    # Drop PII - we only evaluate on the statistical columns
    return df.drop(columns=['Name', 'SSN', 'Email', 'Address'])

def train_and_evaluate_classifier(train_data, test_data, target_col='Default'):
    """
    Trains a Random Forest on `train_data` and evaluates it on `test_data`.
    Returns the accuracy and F1 score.
    """
    # Separate Features (X) and Target (y)
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # Initialize and Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return acc, f1

def main():
    print("--- Starting Evaluation Process ---")
    
    # 1. Load Real Data
    real_data = load_real_data()
    print(f"Loaded Real Data: {real_data.shape}")

    # 2. Load Generator Model
    try:
        synthesizer = CTGANSynthesizer.load('models/financial_synthesizer.pkl')
    except FileNotFoundError:
        print("Error: Model not found. Run 'src/trainer.py' first.")
        return

    # 3. Generate Synthetic Data
    # We generate the same number of rows as the real data for fair comparison
    print("Generating Synthetic Data...")
    synthetic_data = synthesizer.sample(num_rows=len(real_data))
    
    # Save it
    os.makedirs('data/synthetic', exist_ok=True)
    synthetic_data.to_csv('data/synthetic/synthetic_data.csv', index=False)
    print("Synthetic data saved to 'data/synthetic/synthetic_data.csv'")

    # --- SDV QUALITY REPORT ---
    print("\n[1/2] Running SDV Quality Report...")
    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=synthesizer.metadata
    )
    print("Quality Score (0-100):", quality_report.get_score())

    # --- MACHINE LEARNING UTILITY TEST ---
    print("\n[2/2] Running Machine Learning Utility Test...")
    print("Goal: Can a model trained on FAKE data predict REAL defaults?")

    # Split Real Data into Train/Test (80/20)
    # We need a 'Real Test Set' to grade both models fairly.
    real_train, real_test = train_test_split(real_data, test_size=0.2, random_state=42)

    # SCENARIO A: The "Gold Standard" (Train Real -> Test Real)
    real_acc, real_f1 = train_and_evaluate_classifier(real_train, real_test)
    print(f"\n   Baseline (Real Data): Accuracy={real_acc:.4f}, F1-Score={real_f1:.4f}")

    # SCENARIO B: The "Synthetic Experiment" (Train Synthetic -> Test Real)
    # We use the generated synthetic data as the training set.
    # But we MUST test it on the REAL test set to see if it works in the real world.
    syn_acc, syn_f1 = train_and_evaluate_classifier(synthetic_data, real_test)
    print(f"   Synthetic Training  : Accuracy={syn_acc:.4f}, F1-Score={syn_f1:.4f}")

    # --- VERDICT ---
    diff = abs(real_acc - syn_acc)
    print(f"\n--- VERDICT ---")
    print(f"Accuracy Difference: {diff:.4f}")
    if diff < 0.10:
        print("SUCCESS: The synthetic data is high-utility! (Less than 10% performance drop)")
    else:
        print("WARNING: The synthetic data may be losing too much signal. Try training for more epochs.")

if __name__ == "__main__":
    main()
