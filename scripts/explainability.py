import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import numpy as np
import os

def explain_model(best_model, X_train, X_test, y_test, clustering_features, best_model_name):
    """
    MODULE 10: Analyse l'explicabilité du modèle avec LIME et importance des features
    """
    print(f"\n🔍 MODULE 10: EXPLICABILITÉ DU MODÈLE - {best_model_name}")
    print("-" * 60)

    # 10.1 Importance des features (Permutation)
    print("📊 Calcul de l'importance des features...")

    perm_importance = permutation_importance(
        best_model, X_test, y_test, 
        n_repeats=10, random_state=42, scoring='f1_weighted'
    )

    feature_importance_df = pd.DataFrame({
        'feature': clustering_features,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)  # Tri décroissant pour le graphique

    # Création du graphique à barres pour l'importance des features
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df)))
    
    bars = plt.bar(range(len(feature_importance_df)), 
                   feature_importance_df['importance_mean'],
                   yerr=feature_importance_df['importance_std'],
                   color=colors,
                   alpha=0.8,
                   capsize=5)
    
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importance (Permutation)', fontsize=12, fontweight='bold')
    plt.title(f'Importance des Features - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(feature_importance_df)), 
               feature_importance_df['feature'], 
               rotation=45, ha='right')
    
    # Ajout des valeurs sur les barres
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + feature_importance_df.iloc[i]['importance_std'],
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../plots/06_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Importance des features calculée et visualisée")

    # 10.2 Analyse LIME
    print("\n🌿 Analyse LIME...")

    try:
        class_names = ['Faible Risque', 'Haut Risque']

        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=clustering_features,
            class_names=class_names,
            mode="classification"
        )

        # Analyser plusieurs observations pour avoir une vue d'ensemble
        num_samples = min(5, len(X_test))  # Analyser jusqu'à 5 échantillons
        
        all_explanations = []
        
        for i in range(num_samples):
            exp = explainer.explain_instance(
                data_row=X_test.iloc[i].values,
                predict_fn=best_model.predict_proba,
                num_features=len(clustering_features)
            )
            
            # Extraction des données d'explication
            exp_list = exp.as_list()
            all_explanations.append({
                'sample_id': i,
                'explanations': exp_list,
                'prediction': best_model.predict(X_test.iloc[i:i+1])[0],
                'prediction_proba': best_model.predict_proba(X_test.iloc[i:i+1])[0]
            })
        
        # 10.2.1 Graphique global des explications LIME
        print(f"📊 Création des graphiques LIME pour {num_samples} échantillons...")
        
        # Créer un graphique pour chaque échantillon analysé
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, explanation in enumerate(all_explanations):
            if idx >= 5:  # Limiter à 5 graphiques
                break
                
            ax = axes[idx]
            
            # Préparation des données pour le graphique
            features = [item[0] for item in explanation['explanations']]
            importance_values = [item[1] for item in explanation['explanations']]
            
            # Couleurs selon l'impact (positif/négatif)
            colors = ['red' if val < 0 else 'green' for val in importance_values]
            
            # Graphique à barres horizontales
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importance_values, color=colors, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Impact sur la prédiction')
            
            # Titre avec informations sur la prédiction
            pred_class = class_names[explanation['prediction']]
            pred_proba = explanation['prediction_proba'][explanation['prediction']]
            ax.set_title(f'Échantillon {idx+1}\nPrédiction: {pred_class}\nConfiance: {pred_proba:.3f}')
            
            # Ligne verticale à x=0
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            
            # Ajout des valeurs sur les barres
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01 if width >= 0 else width - 0.01, 
                       bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left' if width >= 0 else 'right', 
                       va='center', fontsize=9)
        
        # Supprimer les axes vides
        for idx in range(num_samples, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('../plots/07_lime_explanations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 10.2.2 Graphique d'importance moyenne LIME
        print("📈 Calcul de l'importance moyenne LIME...")
        
        # Agrégation des explications pour toutes les observations
        feature_impacts = {feature: [] for feature in clustering_features}
        
        for explanation in all_explanations:
            for feature_name, impact in explanation['explanations']:
                # Extraire seulement le nom de la feature (sans les conditions)
                clean_feature = feature_name.split(' ')[0]
                if clean_feature in feature_impacts:
                    feature_impacts[clean_feature].append(abs(impact))  # Valeur absolue pour l'importance
        
        # Calcul des moyennes
        avg_importance = {}
        for feature, impacts in feature_impacts.items():
            if impacts:  # Si la liste n'est pas vide
                avg_importance[feature] = np.mean(impacts)
            else:
                avg_importance[feature] = 0
        
        # Tri par importance décroissante
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Graphique de l'importance moyenne LIME
        plt.figure(figsize=(12, 8))
        features_names = [item[0] for item in sorted_features]
        importance_values = [item[1] for item in sorted_features]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(features_names)))
        bars = plt.bar(range(len(features_names)), importance_values, 
                      color=colors, alpha=0.8)
        
        plt.xlabel('Features', fontsize=12, fontweight='bold')
        plt.ylabel('Importance Moyenne LIME (Valeur Absolue)', fontsize=12, fontweight='bold')
        plt.title(f'Importance Moyenne des Features selon LIME - {best_model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(features_names)), features_names, rotation=45, ha='right')
        
        # Ajout des valeurs sur les barres
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('../plots/08_lime_average_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Analyses LIME terminées et visualisées")
        
        # Affichage résumé dans la console
        print("\n📋 RÉSUMÉ DES EXPLICATIONS LIME:")
        print("-" * 40)
        for idx, explanation in enumerate(all_explanations):
            pred_class = class_names[explanation['prediction']]
            pred_proba = explanation['prediction_proba'][explanation['prediction']]
            print(f"Échantillon {idx+1}: {pred_class} (confiance: {pred_proba:.3f})")
            print("  Top 3 features influentes:")
            for j, (feature, impact) in enumerate(explanation['explanations'][:3]):
                direction = "augmente" if impact > 0 else "diminue"
                print(f"    {j+1}. {feature}: {direction} la probabilité ({impact:.4f})")
            print()

    except Exception as e:
        print(f"⚠️ Erreur LIME: {e}")
        print("Analyse LIME non disponible pour ce modèle")

    return feature_importance_df

# ================================================================================================
# FONCTION BONUS: COMPARAISON PERMUTATION vs LIME
# ================================================================================================

def compare_importance_methods(feature_importance_df, lime_avg_importance=None):
    """
    Compare l'importance des features selon différentes méthodes
    """
    if lime_avg_importance is None:
        print("⚠️ Pas de données LIME pour la comparaison")
        return
    
    print("\n🔍 COMPARAISON DES MÉTHODES D'IMPORTANCE")
    print("-" * 60)
    
    # Préparation des données pour la comparaison
    comparison_data = []
    for feature in feature_importance_df['feature']:
        perm_importance = feature_importance_df[feature_importance_df['feature'] == feature]['importance_mean'].iloc[0]
        lime_importance = lime_avg_importance.get(feature, 0)
        comparison_data.append({
            'feature': feature,
            'permutation': perm_importance,
            'lime': lime_importance
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Graphique de comparaison
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, comparison_df['permutation'], width, 
                   label='Permutation', alpha=0.8, color='skyblue')
    bars2 = plt.bar(x + width/2, comparison_df['lime'], width, 
                   label='LIME', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importance', fontsize=12, fontweight='bold')
    plt.title('Comparaison: Importance Permutation vs LIME', fontsize=14, fontweight='bold')
    plt.xticks(x, comparison_df['feature'], rotation=45, ha='right')
    plt.legend()
    
    # Ajout des valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../plots/09_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comparaison des méthodes d'importance visualisée")
    
    return comparison_df