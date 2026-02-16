# 🏦 Synthetic Financial Data Generation (Privacy-Preserving ML)

Hi! 👋 This is a project I built to explore one of the biggest challenges in FinTech: **Data Privacy vs. Data Utility.**

## 🤔 The Problem
Financial institutions have tons of data that could be used to train amazing Machine Learning models (for credit scoring, fraud detection, etc.). But they can't just share this data because it contains **PII** (Personal Identifiable Information) like Names, SSNs, and Addresses.

Masking the data isn't always enough, and it can break the statistical patterns.

## 💡 The Solution
Instead of masking real data, why not generate **Synthetic Data**? 

I used a deep learning model called **CTGAN** (Conditional Tabular GAN) to learn the *patterns* of a dataset without memorizing the *people*. The result is a new dataset that:
1.  Looks mathematically identical to the real data.
2.  Contains **ZERO** real users (100% fake).
3.  Can be used to train machine learning models with high accuracy.

## 🛠️ How It Works
I broke the project down into three main steps:

### 1. The "Fake-Real" Data (`data_generator.py`)
Since I can't upload actual bank data to GitHub, I wrote a script to generate a "Ground Truth" dataset. It creates:
* **Sensitive PII:** Names, SSNs, Emails (using `Faker`).
* **Financial Behaviors:** Income, Credit Score, Loan Default status.
* **Correlations:** I engineered it so that higher income $\rightarrow$ higher credit score $\rightarrow$ lower default chance.

### 2. The Generator (`trainer.py`)
I used the `sdv` library to train a **CTGAN** model. 
* **Input:** The "Fake-Real" data (minus the PII columns).
* **Process:** The model learns the distribution of Income, Credit Scores, and Defaults.
* **Output:** A saved model (`.pkl`) that can generate infinite rows of new data.

### 3. The Utility Test (`evaluator.py`)
This is the cool part. I wanted to prove the synthetic data was actually useful.
1.  I trained a Random Forest on the **Real Data**.
2.  I trained a Random Forest on the **Synthetic Data**.
3.  I tested **BOTH** models on a held-out set of Real Data.

**The Result?** The model trained on synthetic data performed almost as well as the one trained on real data (usually within 5-10% accuracy), proving we don't need to see private data to build good models. 🚀

## 💻 How to Run This

First, clone the repo and install the dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: Generate the "Ground Truth"

```bash
python src/data_generator.py
```

### Step 2: Train the GAN
If you have a GPU, it will auto-detect it. If not, it runs on CPU.

```bash
python src/trainer.py
```

### Step 3: Evaluate the Results
See the comparison between Real vs. Synthetic accuracy.

```bash
python src/evaluator.py
```

📊 Tech Stack
* Python 3.x
* SDV (Synthetic Data Vault): For the CTGAN implementation.
* Faker: To create the realistic PII.
* Scikit-Learn: For the Random Forest evaluation.
* Pandas/Numpy: For data wrangling.

🔜 What's Next?

* Differential Privacy: Adding noise to ensure no single outlier in the real data can be reverse-engineered.
* Multi-Table Data: Generating synthetic data for relational databases (e.g., Users table + Transactions table).

---

Still on maintenance
# synthetic-financial-data
