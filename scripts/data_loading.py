import pandas as pd


def load_and_explore_data(file_path='../data/diabete.csv'):
    """
    MODULE 1: Charge et explore les données initiales du dataset diabète
    
    Cette fonction constitue le point d'entrée du pipeline de données.
    Elle charge le dataset et fournit un aperçu initial de sa structure.
    """
    print("\n📊 MODULE 1: CHARGEMENT ET EXPLORATION DES DONNÉES")
    print("-" * 60)
    
    try:
        df = pd.read_csv(file_path, index_col=0)
        print(f"✅ Dataset chargé: {df.shape[0]} observations, {df.shape[1]} variables")
        
        # Informations sur la structure
        print("\n📋 Structure du dataset:")
        print(df.info())
        
        print("\n👀 Aperçu des premières lignes:")
        print(df.head())
        
        print("\n📈 Statistiques descriptives:")
        print(df.describe())
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier {file_path} non trouvé")
        return None
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None
