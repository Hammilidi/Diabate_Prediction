from fastapi import APIRouter, Request, Form, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from psycopg2.extras import RealDictCursor
from typing import Optional

from utilitaires import (
    get_db_connection, hash_password, verify_password, 
    create_access_token, get_current_user, predict_diabetes_risk,
    ACCESS_TOKEN_EXPIRE_HOURS
)
from model_pydantic import UserCreate, UserLogin, PatientCreate

router = APIRouter()

# Configuration des templates
templates = Jinja2Templates(directory="../templates")

@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {
        "request": request, 
        "error": error
    })

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: str = None):
    return templates.TemplateResponse("register.html", {
        "request": request,
        "error": error
    })

@router.post("/register")
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

@router.post("/login")
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

@router.get("/home", response_class=HTMLResponse)
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

@router.get("/patients", response_class=HTMLResponse)
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

@router.get("/add", response_class=HTMLResponse)
async def add_patient_form(request: Request, token: str = Cookie(None)):
    username = get_current_user(token)
    if not username:
        return RedirectResponse(url="/login", status_code=303)
    
    return templates.TemplateResponse("add_patient.html", {
        "request": request,
        "username": username
    })

@router.post("/submit")
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

@router.get("/delete/{patient_id}")
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

@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("token")
    return response