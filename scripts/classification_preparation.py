import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler




def prepare_classification_data(df_with_clusters, clustering_features):
    """
    MODULE 6: Prépare les données pour la classification supervisée
    
    Cette fonction prépare les features et target, effectue la division
    train/test et gère le déséquilibre des classes.
    """
    print("\n🎯 MODULE 6: PRÉPARATION POUR LA CLASSIFICATION")
    print("-" * 60)
    
    # Définir X et y
    X = df_with_clusters[clustering_features]
    y = df_with_clusters['risk_category']
    
    print(f"📊 Features (X): {X.shape}")
    print(f"🎯 Target (y): {y.shape}")
    print(f"⚖️ Distribution des classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🚂 Training set: {X_train.shape[0]} échantillons")
    print(f"🧪 Test set: {X_test.shape[0]} échantillons")
    
    # Gestion du déséquilibre des classes
    print("\n⚖️ Gestion du déséquilibre des classes...")
    
    # Vérifier le déséquilibre
    class_distribution = y_train.value_counts(normalize=True)
    print("Distribution des classes dans le training set:")
    for class_label, proportion in class_distribution.items():
        risk_level = "Haut risque" if class_label == 1 else "Faible risque"
        print(f"   {risk_level}: {proportion:.3f} ({y_train.value_counts()[class_label]} échantillons)")
    
    # Application de RandomOverSampler si déséquilibre significatif
    if min(class_distribution) < 0.4:  # Si une classe < 40%
        print("🔄 Application de l'over-sampling...")
        oversampler = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)
        
        print(f"✅ Données équilibrées: {X_train_balanced.shape[0]} échantillons")
        print(f"📊 Nouvelle distribution: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}")
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    else:
        print("✅ Classes suffisamment équilibrées, pas de rééchantillonnage nécessaire")
        return X_train, X_test, y_train, y_test
