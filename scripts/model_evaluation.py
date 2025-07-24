import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.model_selection import validation_curve, learning_curve


def evaluate_models(model_results, y_test, trained_models, X_train, y_train, X_test):
    """
    MODULE 8: Évalue et compare les performances des modèles avec analyses avancées
    
    Cette fonction compare tous les modèles, identifie le meilleur et effectue
    des analyses approfondies incluant ROC/AUC et détection d'overfitting.
    """
    print("\n📊 MODULE 8: ÉVALUATION ET COMPARAISON DES MODÈLES")
    print("-" * 60)
    
    # 8.1 Tableau de comparaison des performances
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
    comparison_df.to_csv('../data/model_comparison.csv')
    
    # 8.2 Identification du meilleur modèle
    best_model_name = comparison_df['F1-Score'].idxmax()
    best_f1_score = comparison_df.loc[best_model_name, 'F1-Score']
    best_model = trained_models[best_model_name]
    
    print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
    print(f"🎯 F1-Score: {best_f1_score:.4f}")
    
    # 8.3 Graphiques de comparaison des modèles
    plot_model_comparison(comparison_df)
    
    # 8.4 Analyses détaillées du meilleur modèle
    analyze_best_model(best_model, best_model_name, model_results, y_test, 
                      X_train, y_train, X_test)
    
    return best_model_name, comparison_df


def plot_model_comparison(comparison_df):
    """
    Crée des graphiques de comparaison entre tous les modèles
    """
    print("\n📈 Création des graphiques de comparaison...")
    
    # Graphique à barres des métriques principales
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        values = comparison_df[metric].values
        models = comparison_df.index
        
        bars = ax.bar(range(len(models)), values, color=colors[i], alpha=0.8)
        ax.set_xlabel('Modèles')
        ax.set_ylabel(metric)
        ax.set_title(f'Comparaison - {metric}')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Ajout des valeurs sur les barres
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('../plots/05a_model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Graphique radar pour comparaison globale
    plot_radar_comparison(comparison_df)


def plot_radar_comparison(comparison_df):
    """
    Crée un graphique radar pour comparer les modèles
    """
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Métriques à comparer
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Nombre de variables
    N = len(metrics)
    
    # Angles pour chaque métrique
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Fermer le cercle
    
    # Couleurs pour chaque modèle
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot pour chaque modèle
    for i, (model_name, model_data) in enumerate(comparison_df.iterrows()):
        values = [model_data[metric] for metric in metrics]
        values += values[:1]  # Fermer le cercle
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=model_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # Ajout des labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Comparaison Radar des Modèles', size=16, fontweight='bold', y=1.08)
    plt.savefig('../plots/05b_model_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_best_model(best_model, best_model_name, model_results, y_test, 
                      X_train, y_train, X_test):
    """
    Analyses approfondies du meilleur modèle
    """
    print(f"\n🔍 ANALYSES APPROFONDIES - {best_model_name}")
    print("-" * 60)
    
    best_predictions = model_results[best_model_name]['predictions']
    best_probabilities = model_results[best_model_name]['probabilities']
    
    # 1. Matrice de confusion améliorée
    plot_enhanced_confusion_matrix(y_test, best_predictions, best_model_name)
    
    # 2. Courbes ROC et AUC
    if best_probabilities is not None:
        plot_roc_auc_curves(y_test, best_probabilities, best_model_name)
        plot_precision_recall_curve(y_test, best_probabilities, best_model_name)
    
    # 3. Analyse d'overfitting
    analyze_overfitting(best_model, best_model_name, X_train, y_train, X_test, y_test)
    
    # 4. Courbes d'apprentissage
    plot_learning_curves(best_model, best_model_name, X_train, y_train)
    
    # 5. Rapport de classification détaillé
    print(f"\n📋 Rapport de classification détaillé - {best_model_name}:")
    print(classification_report(y_test, best_predictions, 
                              target_names=['Faible Risque', 'Haut Risque']))


def plot_enhanced_confusion_matrix(y_test, predictions, model_name):
    """
    Matrice de confusion avec pourcentages et statistiques
    """
    cm = confusion_matrix(y_test, predictions)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Matrice de confusion (nombres absolus)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Faible Risque', 'Haut Risque'],
                yticklabels=['Faible Risque', 'Haut Risque'])
    ax1.set_title(f'Matrice de Confusion (Nombres) - {model_name}')
    ax1.set_ylabel('Valeurs Réelles')
    ax1.set_xlabel('Prédictions')
    
    # Matrice de confusion (pourcentages)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                xticklabels=['Faible Risque', 'Haut Risque'],
                yticklabels=['Faible Risque', 'Haut Risque'])
    ax2.set_title(f'Matrice de Confusion (%) - {model_name}')
    ax2.set_ylabel('Valeurs Réelles')
    ax2.set_xlabel('Prédictions')
    
    plt.tight_layout()
    plt.savefig('../plots/05_confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calcul des métriques détaillées
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    print(f"\n📊 Métriques détaillées de la matrice de confusion:")
    print(f"   • Vrais Négatifs (TN): {tn}")
    print(f"   • Faux Positifs (FP): {fp}")
    print(f"   • Faux Négatifs (FN): {fn}")
    print(f"   • Vrais Positifs (TP): {tp}")
    print(f"   • Spécificité: {specificity:.4f}")
    print(f"   • Sensibilité (Recall): {sensitivity:.4f}")


def plot_roc_auc_curves(y_test, probabilities, model_name):
    """
    Courbes ROC et calcul de l'AUC
    """
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Courbe ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Classificateur Aléatoire (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (1 - Spécificité)')
    plt.ylabel('Taux de Vrais Positifs (Sensibilité)')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Subplot 2: Distribution des probabilités
    plt.subplot(1, 2, 2)
    plt.hist(probabilities[y_test == 0], bins=20, alpha=0.7, 
             label='Faible Risque', color='blue', density=True)
    plt.hist(probabilities[y_test == 1], bins=20, alpha=0.7, 
             label='Haut Risque', color='red', density=True)
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Densité')
    plt.title(f'Distribution des Probabilités - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../plots/06_roc_auc_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 AUC Score: {roc_auc:.4f}")
    
    # Analyse du seuil optimal
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"🎯 Seuil optimal: {optimal_threshold:.4f}")
    print(f"   • TPR: {tpr[optimal_idx]:.4f}")
    print(f"   • FPR: {fpr[optimal_idx]:.4f}")


def plot_precision_recall_curve(y_test, probabilities, model_name):
    """
    Courbe Précision-Rappel
    """
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    avg_precision = average_precision_score(y_test, probabilities)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'{model_name} (AP = {avg_precision:.4f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    
    # Ligne de base (proportion de la classe positive)
    baseline = np.sum(y_test) / len(y_test)
    plt.axhline(y=baseline, color='red', linestyle='--', 
                label=f'Baseline (AP = {baseline:.4f})')
    
    plt.xlabel('Rappel (Recall)')
    plt.ylabel('Précision')
    plt.title(f'Courbe Précision-Rappel - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.savefig('../plots/07_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Average Precision Score: {avg_precision:.4f}")


def analyze_overfitting(model, model_name, X_train, y_train, X_test, y_test):
    """
    Analyse d'overfitting avec courbes de validation
    """
    print(f"\n🔍 ANALYSE D'OVERFITTING - {model_name}")
    print("-" * 40)
    
    # Scores sur train vs test
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfitting_gap = train_score - test_score
    
    print(f"📊 Score Train: {train_score:.4f}")
    print(f"📊 Score Test: {test_score:.4f}")
    print(f"📊 Écart (Overfitting): {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.1:
        print("⚠️  ALERTE: Possible overfitting détecté!")
    elif overfitting_gap < 0.05:
        print("✅ Pas d'overfitting significatif")
    else:
        print("⚡ Overfitting modéré")
    
    # Courbes de validation pour différents hyperparamètres
    plot_validation_curves(model, model_name, X_train, y_train)


def plot_validation_curves(model, model_name, X_train, y_train):
    """
    Courbes de validation pour analyser l'overfitting
    """
    # Définir les paramètres à analyser selon le type de modèle
    param_configs = {
        'Random Forest': ('n_estimators', [10, 50, 100, 200, 300, 500]),
        'Decision Tree': ('max_depth', [1, 3, 5, 7, 10, 15, 20, None]),
        'SVM': ('C', [0.001, 0.01, 0.1, 1, 10, 100, 1000]),
        'Gradient Boosting': ('n_estimators', [10, 50, 100, 200, 300, 500]),
        'Logistic Regression': ('C', [0.001, 0.01, 0.1, 1, 10, 100, 1000])
    }
    
    if model_name not in param_configs:
        print(f"⚠️ Courbes de validation non disponibles pour {model_name}")
        return
    
    param_name, param_range = param_configs[model_name]
    
    try:
        train_scores, test_scores = validation_curve(
            model, X_train, y_train, param_name=param_name, 
            param_range=param_range, cv=5, scoring='f1_weighted', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Courbes de validation
        plt.subplot(1, 2, 1)
        plt.plot(param_range, train_mean, 'o-', color='blue', 
                label='Score Train', linewidth=2)
        plt.fill_between(param_range, train_mean - train_std, 
                        train_mean + train_std, alpha=0.2, color='blue')
        
        plt.plot(param_range, test_mean, 'o-', color='red', 
                label='Score Validation', linewidth=2)
        plt.fill_between(param_range, test_mean - test_std, 
                        test_mean + test_std, alpha=0.2, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('Score F1')
        plt.title(f'Courbes de Validation - {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if param_name == 'max_depth' and None in param_range:
            # Remplacer None par une valeur numérique pour l'affichage
            param_range_display = [x if x is not None else max([x for x in param_range if x is not None]) + 5 
                                 for x in param_range]
            plt.xticks(param_range_display, param_range)
        
        # Subplot 2: Écart train-test
        plt.subplot(1, 2, 2)
        gap = train_mean - test_mean
        plt.plot(param_range, gap, 'o-', color='green', linewidth=2)
        plt.axhline(y=0.1, color='red', linestyle='--', 
                   label='Seuil Overfitting (0.1)')
        plt.xlabel(param_name)
        plt.ylabel('Écart Train - Validation')
        plt.title(f'Analyse Overfitting - {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if param_name == 'max_depth' and None in param_range:
            plt.xticks(param_range_display, param_range)
        
        plt.tight_layout()
        plt.savefig('../plots/08_validation_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identification du meilleur paramètre
        best_idx = np.argmax(test_mean)
        best_param = param_range[best_idx]
        best_score = test_mean[best_idx]
        best_gap = gap[best_idx]
        
        print(f"🎯 Meilleur {param_name}: {best_param}")
        print(f"📊 Meilleur score validation: {best_score:.4f}")
        print(f"📊 Écart correspondant: {best_gap:.4f}")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des courbes de validation: {e}")


def plot_learning_curves(model, model_name, X_train, y_train):
    """
    Courbes d'apprentissage pour analyser la performance vs taille du dataset
    """
    print(f"\n📈 COURBES D'APPRENTISSAGE - {model_name}")
    print("-" * 40)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1_weighted'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Courbes d'apprentissage
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, train_mean, 'o-', color='blue', 
             label='Score Train', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, 
                    train_mean + train_std, alpha=0.2, color='blue')
    
    plt.plot(train_sizes, test_mean, 'o-', color='red', 
             label='Score Validation', linewidth=2)
    plt.fill_between(train_sizes, test_mean - test_std, 
                    test_mean + test_std, alpha=0.2, color='red')
    
    plt.xlabel('Taille du Dataset d\'Entraînement')
    plt.ylabel('Score F1')
    plt.title(f'Courbes d\'Apprentissage - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 2: Convergence des scores
    plt.subplot(1, 2, 2)
    gap = train_mean - test_mean
    plt.plot(train_sizes, gap, 'o-', color='green', linewidth=2, 
             label='Écart Train-Validation')
    plt.axhline(y=0.1, color='red', linestyle='--', 
               label='Seuil Overfitting')
    plt.xlabel('Taille du Dataset d\'Entraînement')
    plt.ylabel('Écart Train - Validation')
    plt.title(f'Convergence - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../plots/09_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyse de la convergence
    final_gap = gap[-1]
    if final_gap > 0.1:
        print("⚠️  Le modèle pourrait bénéficier de plus de données")
    else:
        print("✅ Le modèle converge bien avec la taille actuelle du dataset")
    
    print(f"📊 Score final train: {train_mean[-1]:.4f}")
    print(f"📊 Score final validation: {test_mean[-1]:.4f}")
    print(f"📊 Écart final: {final_gap:.4f}")