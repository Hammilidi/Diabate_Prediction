def evaluate_models(model_results, y_test):
    """
    MODULE 8: Évalue et compare les performances des modèles
    
    Cette fonction compare tous les modèles et identifie le meilleur.
    """
    print("\n📊 MODULE 8: ÉVALUATION ET COMPARAISON DES MODÈLES")
    print("-" * 60)
    
    # Tableau de comparaison
    comparison_df = pd.DataFrame({
        model: {
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'CV F1 Mean': results['cv_mean'],
            'CV F1 Std': results['cv_std']
        }
        for model, results in model_results.items()
    }).T
    
    comparison_df = comparison_df.round(4)
    print("📋 Comparaison des performances:")
    print(comparison_df)
    
    # Sauvegarde du tableau de comparaison
    comparison_df.to_csv('models/model_comparison.csv')
    
    # Identification du meilleur modèle
    best_model_name = comparison_df['F1-Score'].idxmax()
    best_f1_score = comparison_df.loc[best_model_name, 'F1-Score']
    
    print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
    print(f"🎯 F1-Score: {best_f1_score:.4f}")
    
    # Matrice de confusion pour le meilleur modèle
    best_predictions = model_results[best_model_name]['predictions']
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Faible Risque', 'Haut Risque'],
                yticklabels=['Faible Risque', 'Haut Risque'])
    plt.title(f'Matrice de Confusion - {best_model_name}')
    plt.ylabel('Valeurs Réelles')
    plt.xlabel('Prédictions')
    plt.savefig('plots/05_confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Rapport de classification détaillé
    print(f"\n📋 Rapport de classification détaillé - {best_model_name}:")
    print(classification_report(y_test, best_predictions, 
                              target_names=['Faible Risque', 'Haut Risque']))
    
    return best_model_name, comparison_df
