from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib
import numpy as np



def perform_clustering(df_scaled, max_clusters=10):
    """
    MODULE 4: Effectue le clustering K-Means avec détermination optimale de k
    
    Cette fonction utilise la méthode du coude ET la méthode des silhouettes
    pour déterminer le nombre optimal de clusters.
    """
    print("\n🎯 MODULE 4: CLUSTERING K-MEANS")
    print("-" * 60)
    
    # 4.1 Méthode du coude pour déterminer k optimal
    print("📈 Détermination du nombre optimal de clusters...")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)  
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(df_scaled)
        inertias.append(kmeans.inertia_)
        
        # Calcul du score de silhouette
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Visualisation combinée: Méthode du coude + Silhouettes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Méthode du coude
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_title('Méthode du Coude')
    ax1.set_xlabel('Nombre de Clusters (k)')
    ax1.set_ylabel('Inertie')
    ax1.grid(True, alpha=0.3)
    
    # Méthode des silhouettes
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_title('Score de Silhouette')
    ax2.set_xlabel('Nombre de Clusters (k)')
    ax2.set_ylabel('Score de Silhouette')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../plots/03_cluster_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Détermination du k optimal
    # Choisir le k avec le meilleur score de silhouette
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    max_silhouette = max(silhouette_scores)
    
    print(f"🎯 Nombre optimal de clusters (Silhouette): k = {optimal_k_silhouette}")
    print(f"📊 Score de silhouette maximal: {max_silhouette:.4f}")
    
    # 4.2 Entraînement du modèle K-Means final
    print(f"\n🤖 Entraînement du modèle K-Means avec k={optimal_k_silhouette}...")
    
    kmeans_final = KMeans(n_clusters=optimal_k_silhouette, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(df_scaled)
    
    print(f"✅ Clustering terminé")
    print(f"📊 Répartition des clusters: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")
    
    # Sauvegarde du modèle de clustering
    joblib.dump({
        'kmeans_model': kmeans_final,
        'optimal_k': optimal_k_silhouette,
        'silhouette_score': max_silhouette
    }, '../models/kmeans_clustering_model.pkl')
    
    print("💾 Modèle de clustering sauvegardé dans models/")
    
    return kmeans_final, cluster_labels, optimal_k_silhouette
