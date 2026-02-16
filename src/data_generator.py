import pandas as pd
import numpy as np
from faker import Faker
import os

# Initialize Faker
fake = Faker()

def generate_fake_real_data(num_rows=5000):
    """
    Generates a dataset that mimics real financial data with PII.
    
    The goal is to create data where:
    1. PII columns (Name, SSN) are unique and sensitive.
    2. Financial columns have statistical relationships (correlations).
       - Higher Income -> Slightly Higher Credit Score
       - Lower Credit Score -> Higher Probability of Default
    """
    print(f"Generating {num_rows} rows of 'fake-real' data...")
    
    data = []

    for _ in range(num_rows):
        # --- 1. SENSITIVE PII (The stuff we must protect) ---
        name = fake.name()
        ssn = fake.ssn()
        address = fake.address().replace('\n', ', ')
        email = fake.email()

        # --- 2. FINANCIAL FEATURES (The patterns we want to learn) ---
        
        # Income: Use Log-Normal distribution (Real income is not a Bell Curve; it has a long tail)
        # mean=11.0, sigma=0.5 results in a median income around $60k but with some millionaires
        income = round(np.random.lognormal(mean=11.0, sigma=0.5), 2)
        
        # Credit Score: correlated with income, but with noise
        # Base score is 650. Higher income adds points. Random noise (+/- 50) added.
        income_factor = (income / 100000) * 40
        random_noise = np.random.normal(0, 40)
        raw_score = 650 + income_factor + random_noise
        credit_score = int(np.clip(raw_score, 300, 850)) # Clip to valid range

        # Debt-to-Income Ratio (DTI): Random float between 0.1 and 0.9
        dti = round(np.random.uniform(0.1, 0.9), 2)

        # --- 3. TARGET VARIABLE (The thing we want to predict) ---
        
        # Loan Default: Depends heavily on Credit Score and DTI
        # We use a logistic function to turn the score into a probability
        # High Score + Low DTI = Low Probability of Default
        
        logit = (credit_score - 600) / 50 - (dti * 2)
        prob_default = 1 / (1 + np.exp(logit)) # Sigmoid function
        
        # Flip a weighted coin based on the probability
        default = np.random.choice([0, 1], p=[1 - prob_default, prob_default])

        data.append([name, ssn, email, address, income, credit_score, dti, default])

    # Create DataFrame
    columns = ['Name', 'SSN', 'Email', 'Address', 'Income', 'CreditScore', 'DTI', 'Default']
    df = pd.DataFrame(data, columns=columns)
    
    # Ensure directory exists
    output_path = 'data/raw/sensitive_financial_data.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Success! Data saved to: {output_path}")
    print(df.head())

if __name__ == "__main__":
    generate_fake_real_data()
