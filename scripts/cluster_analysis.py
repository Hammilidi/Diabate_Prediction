import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def analyze_clusters(df_processed, df_scaled, cluster_labels, clustering_features):
    """
    MODULE 5: Analyse et visualise les clusters obtenus
    
    Cette fonction effectue l'analyse des clusters avec PCA et détermine
    les profils de risque. Les visualisations sont sauvegardées.
    """
    print("\n📊 MODULE 5: ANALYSE DES CLUSTERS")
    print("-" * 60)
    
    # Ajouter les labels de clusters au dataset
    df_with_clusters = df_processed.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # 5.1 Réduction de dimensionnalité avec PCA pour visualisation
    print("🔍 Réduction de dimensionnalité (PCA) pour visualisation...")
    
    pca = PCA(n_components=2, random_state=42)
    df_pca = pca.fit_transform(df_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"✅ Variance expliquée: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}")
    print(f"📊 Variance totale expliquée: {sum(explained_variance):.3f}")
    
    # Visualisation des clusters dans l'espace PCA
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    unique_clusters = np.unique(cluster_labels)
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        plt.scatter(df_pca[mask, 0], df_pca[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'Cluster {cluster}', 
                   alpha=0.6, s=50)
    
    plt.xlabel(f'Première Composante Principale ({explained_variance[0]:.1%} variance)')
    plt.ylabel(f'Deuxième Composante Principale ({explained_variance[1]:.1%} variance)')
    plt.title('Visualisation des Clusters dans l\'Espace PCA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../plots/04_clusters_pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5.2 Analyse des caractéristiques par cluster
    print("\n📋 Analyse des caractéristiques moyennes par cluster:")
    
    cluster_analysis = df_with_clusters.groupby('Cluster')[clustering_features].agg([
        'mean', 'std', 'count'
    ]).round(3)
    
    print(cluster_analysis)
    
    # 5.3 Création de la variable de risque basée sur les clusters
    print("\n🎯 Création de la variable de risque...")
    
    # Analyse pour déterminer quel cluster représente le haut risque
    cluster_means = df_with_clusters.groupby('Cluster')[['Glucose', 'BMI', 'Age']].mean()
    
    # Critères de haut risque: Glucose > 126, BMI > 30, Age élevé
    risk_scores = []
    for cluster in cluster_means.index:
        glucose_risk = 1 if cluster_means.loc[cluster, 'Glucose'] > 126 else 0
        bmi_risk = 1 if cluster_means.loc[cluster, 'BMI'] > 30 else 0
        age_risk = 1 if cluster_means.loc[cluster, 'Age'] > cluster_means['Age'].median() else 0
        
        total_risk = glucose_risk + bmi_risk + age_risk
        risk_scores.append(total_risk)
    
    # Le cluster avec le score de risque le plus élevé
    high_risk_cluster = cluster_means.index[np.argmax(risk_scores)]
    
    print(f"🔴 Cluster à haut risque identifié: Cluster {high_risk_cluster}")
    print(f"📊 Caractéristiques du cluster haut risque:")
    print(f"   - Glucose moyen: {cluster_means.loc[high_risk_cluster, 'Glucose']:.2f}")
    print(f"   - BMI moyen: {cluster_means.loc[high_risk_cluster, 'BMI']:.2f}")
    print(f"   - Âge moyen: {cluster_means.loc[high_risk_cluster, 'Age']:.2f}")
    
    # Création de la variable risk_category
    df_with_clusters['risk_category'] = (df_with_clusters['Cluster'] == high_risk_cluster).astype(int)
    
    risk_distribution = df_with_clusters['risk_category'].value_counts()
    print(f"📊 Distribution du risque: Faible risque={risk_distribution[0]}, Haut risque={risk_distribution[1]}")
    
    return df_with_clusters, pca, high_risk_cluster
