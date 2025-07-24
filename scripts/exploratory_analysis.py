import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def perform_eda(df):
    """
    MODULE 2: Effectue une analyse exploratoire complète des données
    
    Cette fonction analyse la qualité des données, les distributions,
    et les relations entre variables. Les graphiques seront sauvegardés
    dans le dossier plots/.
    """
    print("\n🔍 MODULE 2: ANALYSE EXPLORATOIRE DES DONNÉES")
    print("-" * 60)
    
    # Identification des valeurs manquantes
    missing_data = df.isnull().sum()
    print("❌ Valeurs manquantes par colonne:")
    if missing_data.sum() > 0:
        print(missing_data[missing_data > 0])
    else:
        print("Aucune valeur manquante détectée")
    
    # Identification des doublons
    duplicates = df.duplicated().sum()
    print(f"🔄 Nombre de doublons: {duplicates}")
    
    # Visualisation des distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(df.columns):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution - {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Fréquence')
    
    # Supprimer les subplots vides
    for j in range(len(df.columns), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('../plots/01_distributions_variables.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Matrice de corrélation
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.3f')
    plt.title('Matrice de Corrélation des Variables')
    plt.savefig('../plots/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Graphiques sauvegardés dans plots/")
    
    return correlation_matrix
