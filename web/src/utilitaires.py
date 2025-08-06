import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Optional
import jwt
from dotenv import load_dotenv

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", 24))

# Configuration de la base de données
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# Chargement du modèle ML au démarrage
try:
    model_package = joblib.load('../../models/diabetes_risk_prediction_model_v1.0.pkl')
    ml_model = model_package['model']
    scaler_clf = model_package['classification_scaler']
    features = model_package['features']
    print("✅ Modèle ML chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    ml_model = None
    scaler_clf = None
    features = None

# Connexion à la base de données
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Erreur de connexion à la DB: {e}")
        return None

# Création des tables
def init_database():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        
        # Table médecins
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medecins (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table patients
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
                doctorid INTEGER REFERENCES medecins(id) ON DELETE CASCADE,
                name VARCHAR(100) NOT NULL,
                age INTEGER NOT NULL,
                sex VARCHAR(10) NOT NULL,
                glucose FLOAT NOT NULL,
                bmi FLOAT NOT NULL,
                bloodpressure FLOAT NOT NULL,
                pedigree FLOAT NOT NULL,
                result INTEGER,
                risk_probability FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table prédictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                patientid INTEGER REFERENCES patients(id) ON DELETE CASCADE,
                result INTEGER NOT NULL,
                probability FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Base de données initialisée")

# Fonctions utilitaires
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = None):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except jwt.PyJWTError:
        return None

def predict_diabetes_risk(glucose: float, bmi: float, age: int, pedigree: float):
    """Fonction de prédiction du risque de diabète"""
    if ml_model is None:
        return {"error": "Modèle ML non disponible"}
    
    try:
        # Préparation des données
        data = pd.DataFrame({
            'Glucose': [glucose],
            'BMI': [bmi], 
            'Age': [age],
            'DiabetesPedigreeFunction': [pedigree]
        })
        
        # Standardisation si nécessaire
        if scaler_clf is not None:
            data_scaled = scaler_clf.transform(data)
        else:
            data_scaled = data.values
            
        # Prédiction
        prediction = ml_model.predict(data_scaled)[0]
        probability = ml_model.predict_proba(data_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probability_low_risk': float(probability[0]),
            'probability_high_risk': float(probability[1]),
            'risk_level': 'Haut Risque' if prediction == 1 else 'Faible Risque'
        }
    except Exception as e:
        return {"error": f"Erreur lors de la prédiction: {str(e)}"}