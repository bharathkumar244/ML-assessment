import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def load_and_clean_data(filepath):
    """
    Load and clean the diabetic dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def handle_missing_values_medical(df):
    """
    Handle missing values with medical context awareness
    """
    print("Handling missing values with medical context...")
    df_clean = df.copy()
    
    # Medical context imputation
    df_clean['weight'] = df_clean['weight'].fillna('unknown')
    df_clean['payer_code'] = df_clean['payer_code'].fillna('Unknown')
    df_clean['medical_specialty'] = df_clean['medical_specialty'].fillna('Unknown')
    df_clean['race'] = df_clean['race'].fillna('Caucasian')
    
    # Diagnosis codes - create unknown category
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Create binary target for 30-day readmission
    df_clean['readmission_binary'] = (df_clean['readmitted'] == '<30').astype(int)
    
    print("Missing values handled successfully!")
    return df_clean

def prepare_features_target(df, target_col='readmission_binary'):
    """
    Prepare features and target for modeling
    """
    # Remove columns that are not useful for prediction
    cols_to_drop = ['encounter_id', 'patient_nbr', 'readmitted', 'weight', 
                   'payer_code', 'medical_specialty']
    
    feature_df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Separate features and target
    X = feature_df.drop(target_col, axis=1)
    y = feature_df[target_col]
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Features prepared: {X.shape[1]} total features")
    print(f"  - Categorical: {len(categorical_cols)}")
    print(f"  - Numerical: {len(numerical_cols)}")
    
    return X, y, categorical_cols, numerical_cols