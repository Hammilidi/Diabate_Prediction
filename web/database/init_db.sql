-- init_db.sql
-- Script d'initialisation de la base de données PostgreSQL

-- Créer la base de données
CREATE DATABASE diabetoweb;

-- Table des médecins
CREATE TABLE medecins (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des patients
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    doctorid INTEGER REFERENCES medecins(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL CHECK (age > 0 AND age < 150),
    sex VARCHAR(10) NOT NULL CHECK (sex IN ('M', 'F')),
    glucose FLOAT NOT NULL CHECK (glucose >= 0),
    bmi FLOAT NOT NULL CHECK (bmi >= 10 AND bmi <= 70),
    bloodpressure FLOAT NOT NULL CHECK (bloodpressure >= 50 AND bloodpressure <= 250),
    pedigree FLOAT NOT NULL CHECK (pedigree >= 0 AND pedigree <= 2),
    result INTEGER CHECK (result IN (0, 1)),
    risk_probability FLOAT CHECK (risk_probability >= 0 AND risk_probability <= 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des prédictions
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    patientid INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    result INTEGER NOT NULL CHECK (result IN (0, 1)),
    probability FLOAT NOT NULL CHECK (probability >= 0 AND probability <= 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour améliorer les performances
CREATE INDEX idx_patients_doctorid ON patients(doctorid);
CREATE INDEX idx_patients_created_at ON patients(created_at);
CREATE INDEX idx_predictions_patientid ON predictions(patientid);

-- Insertion de données de test (optionnel)
INSERT INTO medecins (username, email, password) VALUES 
('admin', 'admin@diabetoweb.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeVMstm.j6V4W0K.W'); -- password: admin123

-- Commentaires et documentation
COMMENT ON TABLE medecins IS 'Table des médecins utilisateurs de l\'application';
COMMENT ON TABLE patients IS 'Table des patients avec leurs données médicales';
COMMENT ON TABLE predictions IS 'Table des prédictions IA pour chaque patient';