import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import re
import joblib
import os

# Define the file path for the dataset
DATASET_PATH = 'reply_classification_dataset.csv'

def preprocess_text(text):
    """Cleans and preprocesses the text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = text.strip()
    return text

def train_and_evaluate_model():
    """
    Main function to load data, train a model, and evaluate its performance.
    """
    if not os.path.exists(DATASET_PATH):
        print(f"Error: The dataset file '{DATASET_PATH}' was not found.")
        return

    print("--- Part A: ML/NLP Pipeline ---")
    
    # 1. Load and Preprocess the Dataset
    print("Loading and preprocessing data...")
    df = pd.read_csv(DATASET_PATH)
    
    # Handle missing values
    df.dropna(subset=['reply', 'label'], inplace=True)
    
    # Standardize and clean the 'label' column
    df['label'] = df['label'].str.lower().str.strip()
    
    # Clean the 'reply' text
    df['reply'] = df['reply'].apply(preprocess_text)
    
    # Prepare data for model training
    X = df['reply']
    y = df['label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split complete.")
    
    # 2. Train a Baseline Model (Logistic Regression)
    print("Training a baseline Logistic Regression model...")
    vectorizer = TfidfVectorizer()
    
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(X_train_vectorized, y_train)
    print("Model training complete.")
    
    # 3. Evaluate the Model
    print("Evaluating the model...")
    y_pred = log_reg_model.predict(X_test_vectorized)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 4. Save the Model and Vectorizer
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(log_reg_model, 'model.joblib')
    print("\nVectorizer and trained model saved as 'vectorizer.joblib' and 'model.joblib'.")

if __name__ == '__main__':
    train_and_evaluate_model()