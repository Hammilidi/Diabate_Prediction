import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def preprocess_data(df):
    """
    MODULE 3: Prétraite les données: gestion des valeurs manquantes, outliers, standardisation
    
    Cette fonction nettoie les données et les prépare pour l'analyse.
    Elle gère les valeurs manquantes, supprime les outliers et standardise les variables.
    """
    print("\n🔧 MODULE 3: PRÉTRAITEMENT DES DONNÉES")
    print("-" * 60)
    
    df_processed = df.copy()
    
    # 3.1 Gestion des valeurs manquantes
    print("🔄 Traitement des valeurs manquantes...")
    
    # Imputation par la médiane pour les variables numériques
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"   ✅ {col}: {df[col].isnull().sum()} valeurs imputées par la médiane ({median_val:.2f})")
    
    # 3.2 Détection et suppression des outliers
    print("\n🎯 Détection des valeurs aberrantes...")
    
    initial_shape = df_processed.shape[0]
    
    # Méthode IQR pour chaque variable numérique
    for col in df_processed.select_dtypes(include=[np.number]).columns:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            print(f"   🔍 {col}: {outliers_count} outliers détectés")
            df_processed = df_processed[~outliers_mask]
    
    final_shape = df_processed.shape[0]
    print(f"✅ Outliers supprimés: {initial_shape - final_shape} observations ({((initial_shape - final_shape)/initial_shape)*100:.1f}%)")
    print(f"📊 Dataset final: {final_shape} observations")
    
    # 3.3 Sélection des variables pour le clustering
    clustering_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    print(f"\n🎯 Variables sélectionnées pour le clustering: {clustering_features}")
    
    df_clustering = df_processed[clustering_features].copy()
    
    # 3.4 Standardisation des variables
    print("\n⚖️ Standardisation des variables...")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_clustering),
        columns=clustering_features,
        index=df_clustering.index
    )
    
    print("✅ Standardisation terminée")
    
    return df_processed, df_scaled, scaler, clustering_features
