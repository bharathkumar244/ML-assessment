import pandas as pd
import numpy as np

def create_medical_features(df):
    """
    Create clinically relevant features for readmission prediction
    """
    print("Creating medical features...")
    df_featured = df.copy()
    
    # Age to numerical (medical relevance)
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df_featured['age_numeric'] = df_featured['age'].map(age_mapping)
    
    # Number of medications (polypharmacy risk)
    medication_cols = [col for col in df.columns if 'med' in col.lower() and 'change' not in col.lower()]
    df_featured['num_medications'] = df_featured[medication_cols].apply(
        lambda x: (x != 'No').sum(), axis=1
    )
    
    # Number of diagnoses (comorbidity index)
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    df_featured['num_diagnoses'] = df_featured[diag_cols].apply(
        lambda x: (x != 'Unknown').sum(), axis=1
    )
    
    # Hospital utilization features
    df_featured['number_emergency'] = pd.to_numeric(df_featured['number_emergency'], errors='coerce')
    df_featured['number_inpatient'] = pd.to_numeric(df_featured['number_inpatient'], errors='coerce')
    df_featured['number_outpatient'] = pd.to_numeric(df_featured['number_outpatient'], errors='coerce')
    
    # Create comorbidity flags based on diagnosis codes
    def has_condition(diag_code, prefixes):
        if diag_code == 'Unknown':
            return 0
        try:
            return 1 if any(diag_code.startswith(prefix) for prefix in prefixes) else 0
        except:
            return 0
    
    # Common comorbidities
    df_featured['has_cardiovascular'] = df_featured['diag_1'].apply(
        lambda x: has_condition(x, ['39', '40', '41', '42', '43', '44'])
    )
    df_featured['has_diabetes'] = df_featured['diag_1'].apply(
        lambda x: has_condition(x, ['250'])
    )
    df_featured['has_renal'] = df_featured['diag_1'].apply(
        lambda x: has_condition(x, ['58', 'N17', 'N18', 'N19'])
    )
    
    # Time in hospital as risk factor
    df_featured['time_in_hospital'] = pd.to_numeric(df_featured['time_in_hospital'], errors='coerce')
    
    # A1C result as numerical
    a1c_mapping = {'None': 0, 'Norm': 5, '>7': 8, '>8': 9}
    df_featured['a1c_numeric'] = df_featured['A1Cresult'].map(a1c_mapping)
    
    # Number of lab procedures
    df_featured['num_lab_procedures'] = pd.to_numeric(df_featured['num_lab_procedures'], errors='coerce')
    
    # Number of procedures
    df_featured['num_procedures'] = pd.to_numeric(df_featured['num_procedures'], errors='coerce')
    
    # Number of medications changed
    df_featured['num_medications_changed'] = pd.to_numeric(df_featured['num_medications_changed'], errors='coerce')
    
    print(f"Created {len([col for col in df_featured.columns if col not in df.columns])} new medical features")
    
    return df_featured

def encode_categorical_features(X, categorical_cols):
    """
    Encode categorical features for modeling
    """
    from sklearn.preprocessing import LabelEncoder
    
    X_encoded = X.copy()
    
    # Label encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        if col in X_encoded.columns:
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    
    # Handle any remaining missing values with median
    X_encoded = X_encoded.fillna(X_encoded.median())
    
    return X_encoded

def select_clinical_features(X_encoded, y, method='random_forest', n_features=30):
    """
    Select most clinically relevant features
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    
    if method == 'random_forest':
        selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        selector.fit(X_encoded, y)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        feature_selector = SelectFromModel(selector, prefit=True, max_features=n_features)
        selected_features = X_encoded.columns[feature_selector.get_support()].tolist()
        
        print(f"Selected {len(selected_features)} features using Random Forest")
        
        return selected_features, feature_importance
    
    else:
        # Return all features if no selection method specified
        return X_encoded.columns.tolist(), None