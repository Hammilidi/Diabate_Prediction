
def run_complete_pipeline(data_path='../data/diabete.csv'):
    """
    MODULE 13: Pipeline principal qui exécute tout le processus
    
    Cette fonction orchestre l'exécution de tous les modules.
    """
    print("🚀 DÉMARRAGE DU PIPELINE COMPLET")
    print("="*80)
    
    try:
        # MODULE 1: Chargement des données
        df = load_and_explore_data(data_path)
        if df is None:
            return None
        
        # MODULE 2: Analyse exploratoire
        correlation_matrix = perform_eda(df)
        
        # MODULE 3: Prétraitement
        df_processed, df_scaled, scaler_clustering, clustering_features = preprocess_data(df)
        
        # MODULE 4: Clustering
        kmeans_model, cluster_labels, optimal_k = perform_clustering(df_scaled)
        
        # MODULE 5: Analyse des clusters
        df_with_clusters, pca, high_risk_cluster = analyze_clusters(
            df_processed, df_scaled, cluster_labels, clustering_features
        )
        
        # MODULE 6: Préparation classification
        X_train, X_test, y_train, y_test = prepare_classification_data(
            df_with_clusters, clustering_features
        )
        
        # MODULE 7: Entraînement des modèles
        model_results, trained_models, scaler_clf = train_classification_models(
            X_train, X_test, y_train, y_test
        )
        
        # MODULE 8: Évaluation
        best_model_name, comparison_df = evaluate_models(model_results, y_test)
        
        # MODULE 9: Optimisation
        optimized_model, scaler_opt = optimize_best_model(
            best_model_name, trained_models, X_train, y_train
        )
        
        # MODULE 10: Explicabilité
        feature_importance = explain_model(
            optimized_model, X_train, X_test, y_test, 
            clustering_features, best_model_name
        )
        
        # MODULE 11: Sauvegarde finale
        final_scaler = scaler_opt if scaler_opt is not None else scaler_clf
        model_package = save_final_model(
            optimized_model, final_scaler, scaler_clustering,
            clustering_features, high_risk_cluster, best_model_name
        )
        
        print("\n" + "="*80)
        print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
        print("="*80)
        print(f"🏆 Meilleur modèle: {best_model_name}")
        print(f"📊 Nombre de clusters optimaux: {optimal_k}")
        print(f"🔴 Cluster haut risque: {high_risk_cluster}")
        print(f"💾 Modèle sauvegardé: diabetes_risk_prediction_model_v1.0.pkl")
        print("📁 Graphiques disponibles dans: plots/")
        print("🤖 Modèles disponibles dans: models/")
        
        return model_package
        
    except Exception as e:
        print(f"❌ ERREUR DANS LE PIPELINE: {e}")
        return None

# ================================================================================================
# FONCTION DE DÉMONSTRATION
# ================================================================================================

def demo_prediction():
    """
    Fonction de démonstration des prédictions
    """
    print("\n🎭 DÉMONSTRATION DES PRÉDICTIONS")
    print("-" * 60)
    
    # Exemples de patients
    patients = [
        {"name": "Patient A (Faible risque)", "glucose": 85, "bmi": 23, "age": 25, "pedigree": 0.2},
        {"name": "Patient B (Risque modéré)", "glucose": 120, "bmi": 28, "age": 45, "pedigree": 0.5},
        {"name": "Patient C (Haut risque)", "glucose": 180, "bmi": 35, "age": 60, "pedigree": 0.8}
    ]
    
    for patient in patients:
        print(f"\n👤 {patient['name']}")
        result = predict_diabetes_risk(
            patient['glucose'], patient['bmi'], 
            patient['age'], patient['pedigree']
        )
        
        if 'error' not in result:
            print(f"   🎯 Prédiction: {result['risk_level']}")
            print(f"   📊 Confiance: {result['confidence']:.1f}%")
            print(f"   🔴 Probabilité haut risque: {result['probability_high_risk']:.1f}%")
        else:
            print(f"   ❌ {result['error']}")

# ================================================================================================
# POINT D'ENTRÉE PRINCIPAL
# ================================================================================================

if __name__ == "__main__":
    # Exécution du pipeline complet
    model_package = run_complete_pipeline()
    
    if model_package is not None:
        # Démonstration des prédictions
        demo_prediction()
        
        print("\n" + "="*80)
        print("📚 POUR UTILISER LE MODÈLE:")
        print("="*80)
        print("# Exemple d'utilisation:")
        print("result = predict_diabetes_risk(glucose=140, bmi=32, age=50, diabetes_pedigree=0.6)")
        print("print(result)")
        print("\n💡 Le système est maintenant prêt pour la production!")
    else:
        print("❌ Échec du pipeline. Veuillez vérifier les données d'entrée.")