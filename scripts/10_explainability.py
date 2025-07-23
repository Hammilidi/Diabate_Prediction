def explain_model(best_model, X_train, X_test, y_test, clustering_features, best_model_name):
    """
    MODULE 10: Analyse l'explicabilité du modèle avec SHAP et importance des features
    
    Cette fonction fournit des explications sur les prédictions du modèle.
    """
    print(f"\n🔍 MODULE 10: EXPLICABILITÉ DU MODÈLE - {best_model_name}")
    print("-" * 60)
    
    # 10.1 Importance des features (permutation)
    print("📊 Calcul de l'importance des features...")
    
    perm_importance = permutation_importance(
        best_model, X_test, y_test, 
        n_repeats=10, random_state=42, scoring='f1_weighted'
    )
    
    # Visualisation de l'importance des features
    feature_importance_df = pd.DataFrame({
        'feature': clustering_features,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance_mean'], 
             xerr=feature_importance_df['importance_std'])
    plt.xlabel('Importance (Permutation)')
    plt.title(f'Importance des Features - {best_model_name}')
    plt.tight_layout()
    plt.savefig('plots/06_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Importance des features calculée et visualisée")
    
    # 10.2 Analyse SHAP (si applicable)
    print("\n🎯 Analyse SHAP...")
    
    try:
        if best_model_name == 'Random Forest':
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)
            
            # Si classification binaire, prendre les valeurs pour la classe positive
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
        elif best_model_name in ['SVM', 'Logistic Regression']:
            explainer = shap.LinearExplainer(best_model, X_train)
            shap_values = explainer.shap_values(X_test)
            
        else:
            explainer = shap.KernelExplainer(best_model.predict_proba, X_train[:100])
            shap_values = explainer.shap_values(X_test[:50])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Graphique SHAP summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=clustering_features, show=False)
        plt.savefig('plots/07_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Analyse SHAP terminée")
        
    except Exception as e:
        print(f"⚠️ Erreur SHAP: {e}")
        print("Analyse SHAP non disponible pour ce modèle")
    
    return feature_importance_df
