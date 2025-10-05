import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

def split_data_patient_level(df, X_encoded, y, test_size=0.3, patient_col='patient_nbr', random_state=42):
    """
    Split data by patient to prevent temporal leakage
    """
    # Get unique patients
    unique_patients = df[patient_col].unique()
    
    # Split patients
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state, stratify=None
    )
    
    # Create masks
    train_mask = df[patient_col].isin(train_patients)
    test_mask = df[patient_col].isin(test_patients)
    
    # Split data
    X_train = X_encoded.loc[train_mask]
    X_test = X_encoded.loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
    
    print(f"Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples ({len(train_patients)} unique patients)")
    print(f"  Test set: {X_test.shape[0]} samples ({len(test_patients)} unique patients)")
    print(f"  Train target distribution: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
    print(f"  Test target distribution: 0={sum(y_test==0)}, 1={sum(y_test==1)}")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple ML models for comparison
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'SVM': SVC(probability=True, random_state=42, kernel='rbf')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'SVM':
            # Scale data for SVM
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        brier = brier_score_loss(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'brier_score': brier,
            'auc_score': auc_score
        }
        
        print(f"  Brier Score: {brier:.4f}")
        print(f"  AUC Score: {auc_score:.4f}")
    
    return results

def calibrate_models(results, X_train, X_test, y_train, y_test):
    """
    Calibrate models to improve probability estimates
    """
    calibrated_results = {}
    
    for name, result in results.items():
        print(f"Calibrating {name}...")
        
        model = result['model']
        
        if name == 'SVM':
            # For SVM, scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated_model.fit(X_train_scaled, y_train)
            y_pred_proba_calibrated = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        else:
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated_model.fit(X_train, y_train)
            y_pred_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate calibrated Brier score
        brier_calibrated = brier_score_loss(y_test, y_pred_proba_calibrated)
        
        calibrated_results[name] = {
            'calibrated_model': calibrated_model,
            'y_pred_proba_calibrated': y_pred_proba_calibrated,
            'brier_score_calibrated': brier_calibrated
        }
        
        print(f"  Original Brier: {result['brier_score']:.4f}")
        print(f"  Calibrated Brier: {brier_calibrated:.4f}")
        print(f"  Improvement: {result['brier_score'] - brier_calibrated:.4f}")
    
    return calibrated_results

def save_trained_models(results, calibrated_results, output_dir='models'):
    """
    Save trained models to disk
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name in results.keys():
        # Save original model
        joblib.dump(results[name]['model'], f'{output_dir}/{name.lower().replace(" ", "_")}_model.pkl')
        
        # Save calibrated model if available
        if name in calibrated_results:
            joblib.dump(calibrated_results[name]['calibrated_model'], 
                       f'{output_dir}/{name.lower().replace(" ", "_")}_calibrated_model.pkl')
    
    print(f"All models saved to {output_dir}/")

def load_trained_models(model_names, output_dir='models'):
    """
    Load trained models from disk
    """
    models = {}
    
    for name in model_names:
        model_path = f'{output_dir}/{name.lower().replace(" ", "_")}_calibrated_model.pkl'
        try:
            models[name] = joblib.load(model_path)
            print(f"Loaded {name} model")
        except:
            model_path = f'{output_dir}/{name.lower().replace(" ", "_")}_model.pkl'
            models[name] = joblib.load(model_path)
            print(f"Loaded {name} model (original)")
    
    return models