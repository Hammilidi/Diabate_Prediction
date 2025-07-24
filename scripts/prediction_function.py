import joblib
import pandas as pd


def predict_diabetes_risk(glucose, bmi, age, diabetes_pedigree, model_path=None):
    """
    MODULE 12: Fonction de prédiction pour nouveaux patients
    
    Cette fonction charge le modèle sauvegardé et effectue des prédictions.
    """
    if model_path is None:
        model_path = '../models/diabetes_risk_prediction_model_v1.0.pkl'
    
    try:
        # Chargement du modèle
        model_package = joblib.load(model_path)
        
        # Extraction des composants
        model = model_package['model']
        scaler = model_package['classification_scaler']
        features = model_package['features']
        
        # Préparation des données d'entrée
        input_data = pd.DataFrame({
            'Glucose': [glucose],
            'BMI': [bmi],
            'Age': [age],
            'DiabetesPedigreeFunction': [diabetes_pedigree]
        })
        
        # Prédiction
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
        else:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        # Interprétation
        risk_level = "HAUT RISQUE" if prediction == 1 else "FAIBLE RISQUE"
        confidence = max(probability) * 100
        
        result = {
            'prediction': int(prediction),
            'risk_level': risk_level,
            'confidence': confidence,
            'probability_low_risk': probability[0] * 100,
            'probability_high_risk': probability[1] * 100,
            'input_values': {
                'Glucose': glucose,
                'BMI': bmi,
                'Age': age,
                'DiabetesPedigreeFunction': diabetes_pedigree
            }
        }
        
        return result
        
    except Exception as e:
        return {'error': f'Erreur lors de la prédiction: {e}'}
