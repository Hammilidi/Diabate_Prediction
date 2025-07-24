import joblib
from datetime import datetime


def save_final_model(best_model, scaler_clf, scaler_clustering, clustering_features, 
                    high_risk_cluster, best_model_name):
    """
    MODULE 11: Sauvegarde le modèle final pour déploiement
    
    Cette fonction crée un package complet pour la mise en production.
    """
    print(f"\n💾 MODULE 11: SAUVEGARDE DU MODÈLE FINAL")
    print("-" * 60)
    
    # Package complet du modèle
    model_package = {
        'model': best_model,
        'classification_scaler': scaler_clf,
        'clustering_scaler': scaler_clustering,
        'features': clustering_features,
        'high_risk_cluster': high_risk_cluster,
        'model_name': best_model_name,
        'version': '1.0',
        'created_date': datetime.now().isoformat(),
        'description': 'Modèle de prédiction du risque de diabète basé sur clustering K-Means'
    }
    
    # Sauvegarde
    joblib.dump(model_package, '../models/diabetes_risk_prediction_model_v1.0.pkl')
    
    print("✅ Modèle final sauvegardé: models/diabetes_risk_prediction_model_v1.0.pkl")
    print(f"🎯 Modèle: {best_model_name}")
    print(f"📊 Features: {clustering_features}")
    print(f"📅 Date de création: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model_package
