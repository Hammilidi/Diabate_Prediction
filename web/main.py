from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
from pydantic import BaseModel
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# Configuration
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Configuration de la base de données
DB_CONFIG = {
    "host": "localhost",
    "database": "diabetoweb",
    "user": "postgres",
    "password": "admin"
}

app = FastAPI(title="DiabetoWeb", description="Application de prédiction du diabète")

# Configuration des templates et fichiers statiques
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Modèles Pydantic
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str

class UserLogin(BaseModel):
    username: str
    password: str

class PatientCreate(BaseModel):
    name: str
    age: int
    sex: str
    glucose: float
    bmi: float
    bloodpressure: float
    pedigree: float

# Chargement du modèle ML au démarrage
try:
    model_package = joblib.load('../models/diabetes_risk_prediction_model_v1.0.pkl')
    ml_model = model_package['model']
    scaler_clf = model_package['classification_scaler']
    features = model_package['features']
    print("✅ Modèle ML chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    ml_model = None

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

def get_current_user(token: str = Cookie(None)):
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

# Routes
@app.on_event("startup")
async def startup_event():
    init_database()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {
        "request": request, 
        "error": error
    })

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: str = None):
    return templates.TemplateResponse("register.html", {
        "request": request,
        "error": error
    })

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Les mots de passe ne correspondent pas"
        })
    
    conn = get_db_connection()
    if not conn:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Erreur de base de données"
        })
    
    cursor = conn.cursor()
    
    # Vérifier si l'utilisateur existe déjà
    cursor.execute("SELECT id FROM medecins WHERE username = %s OR email = %s", (username, email))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Nom d'utilisateur ou email déjà utilisé"
        })
    
    # Créer le nouvel utilisateur
    hashed_password = hash_password(password)
    cursor.execute(
        "INSERT INTO medecins (username, email, password) VALUES (%s, %s, %s)",
        (username, email, hashed_password)
    )
    conn.commit()
    cursor.close()
    conn.close()
    
    return RedirectResponse(url="/login?success=Compte créé avec succès", status_code=303)

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    conn = get_db_connection()
    if not conn:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Erreur de base de données"
        })
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM medecins WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not user or not verify_password(password, user['password']):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Nom d'utilisateur ou mot de passe incorrect"
        })
    
    # Créer le token
    access_token = create_access_token(data={"sub": user['username'], "user_id": user['id']})
    
    response = RedirectResponse(url="/home", status_code=303)
    response.set_cookie(
        key="token", 
        value=access_token, 
        httponly=True, 
        max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600
    )
    return response

@app.get("/home", response_class=HTMLResponse)
async def home(request: Request, token: str = Cookie(None)):
    username = get_current_user(token)
    if not username:
        return RedirectResponse(url="/login", status_code=303)
    
    # Récupérer les statistiques
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Récupérer l'ID du médecin
        cursor.execute("SELECT id FROM medecins WHERE username = %s", (username,))
        doctor = cursor.fetchone()
        doctor_id = doctor['id'] if doctor else None
        
        # Statistiques générales
        cursor.execute("SELECT COUNT(*) as total FROM patients WHERE doctorid = %s", (doctor_id,))
        total_patients = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as diabetic FROM patients WHERE doctorid = %s AND result = 1", (doctor_id,))
        diabetic_patients = cursor.fetchone()['diabetic']
        
        # Patients récents
        cursor.execute("""
            SELECT name, age, result, risk_probability, created_at 
            FROM patients 
            WHERE doctorid = %s 
            ORDER BY created_at DESC 
            LIMIT 5
        """, (doctor_id,))
        recent_patients = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        diabetic_percentage = (diabetic_patients / total_patients * 100) if total_patients > 0 else 0
    else:
        total_patients = 0
        diabetic_patients = 0
        diabetic_percentage = 0
        recent_patients = []
    
    return templates.TemplateResponse("home.html", {
        "request": request,
        "username": username,
        "total_patients": total_patients,
        "diabetic_patients": diabetic_patients,
        "diabetic_percentage": round(diabetic_percentage, 1),
        "recent_patients": recent_patients
    })

@app.get("/patients", response_class=HTMLResponse)
async def patients_list(request: Request, token: str = Cookie(None)):
    username = get_current_user(token)
    if not username:
        return RedirectResponse(url="/login", status_code=303)
    
    conn = get_db_connection()
    if not conn:
        return templates.TemplateResponse("patients.html", {
            "request": request,
            "patients": [],
            "error": "Erreur de base de données"
        })
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Récupérer l'ID du médecin
    cursor.execute("SELECT id FROM medecins WHERE username = %s", (username,))
    doctor = cursor.fetchone()
    doctor_id = doctor['id'] if doctor else None
    
    # Récupérer tous les patients
    cursor.execute("""
        SELECT id, name, age, sex, glucose, bmi, bloodpressure, pedigree, 
               result, risk_probability, created_at
        FROM patients 
        WHERE doctorid = %s 
        ORDER BY created_at DESC
    """, (doctor_id,))
    patients = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return templates.TemplateResponse("patients.html", {
        "request": request,
        "patients": patients,
        "username": username
    })

@app.get("/add", response_class=HTMLResponse)
async def add_patient_form(request: Request, token: str = Cookie(None)):
    username = get_current_user(token)
    if not username:
        return RedirectResponse(url="/login", status_code=303)
    
    return templates.TemplateResponse("add_patient.html", {
        "request": request,
        "username": username
    })

@app.post("/submit")
async def submit_patient(
    request: Request,
    name: str = Form(...),
    age: int = Form(...),
    sex: str = Form(...),
    glucose: float = Form(...),
    bmi: float = Form(...),
    bloodpressure: float = Form(...),
    pedigree: float = Form(...),
    token: str = Cookie(None)
):
    username = get_current_user(token)
    if not username:
        return RedirectResponse(url="/login", status_code=303)
    
    # Prédiction du diabète
    prediction_result = predict_diabetes_risk(glucose, bmi, age, pedigree)
    
    if "error" in prediction_result:
        return templates.TemplateResponse("add_patient.html", {
            "request": request,
            "username": username,
            "error": prediction_result["error"]
        })
    
    conn = get_db_connection()
    if not conn:
        return templates.TemplateResponse("add_patient.html", {
            "request": request,
            "username": username,
            "error": "Erreur de base de données"
        })
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Récupérer l'ID du médecin
    cursor.execute("SELECT id FROM medecins WHERE username = %s", (username,))
    doctor = cursor.fetchone()
    doctor_id = doctor['id'] if doctor else None
    
    # Insérer le patient
    cursor.execute("""
        INSERT INTO patients (doctorid, name, age, sex, glucose, bmi, bloodpressure, pedigree, result, risk_probability)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (doctor_id, name, age, sex, glucose, bmi, bloodpressure, pedigree, 
          prediction_result['prediction'], prediction_result['probability_high_risk']))
    
    patient_id = cursor.fetchone()['id']
    
    # Insérer la prédiction
    cursor.execute("""
        INSERT INTO predictions (patientid, result, probability)
        VALUES (%s, %s, %s)
    """, (patient_id, prediction_result['prediction'], prediction_result['probability_high_risk']))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return templates.TemplateResponse("add_patient.html", {
        "request": request,
        "username": username,
        "success": f"Patient {name} ajouté avec succès!",
        "prediction": prediction_result
    })

@app.get("/delete/{patient_id}")
async def delete_patient(patient_id: int, token: str = Cookie(None)):
    username = get_current_user(token)
    if not username:
        return RedirectResponse(url="/login", status_code=303)
    
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        
        # Récupérer l'ID du médecin
        cursor.execute("SELECT id FROM medecins WHERE username = %s", (username,))
        doctor = cursor.fetchone()
        doctor_id = doctor[0] if doctor else None
        
        # Supprimer le patient (s'assurer qu'il appartient au médecin connecté)
        cursor.execute("DELETE FROM patients WHERE id = %s AND doctorid = %s", (patient_id, doctor_id))
        conn.commit()
        cursor.close()
        conn.close()
    
    return RedirectResponse(url="/patients", status_code=303)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("token")
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)