from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import cross_val_score




def train_classification_models(X_train, X_test, y_train, y_test):
    """
    MODULE 7: Entraîne et évalue plusieurs modèles de classification
    
    Cette fonction entraîne 5 algorithmes différents et compare leurs performances.
    """
    print("\n🤖 MODULE 7: ENTRAÎNEMENT DES MODÈLES DE CLASSIFICATION")
    print("-" * 60)
    
    # Standardisation pour les modèles qui en ont besoin
    scaler_clf = StandardScaler()
    X_train_scaled = scaler_clf.fit_transform(X_train)
    X_test_scaled = scaler_clf.transform(X_test)
    
    # Définition des modèles
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Stockage des résultats
    model_results = {}
    trained_models = {}
    
    print("🏋️ Entraînement des modèles...")
    
    for name, model in models.items():
        print(f"\n   🔄 Entraînement: {name}")
        
        # Utiliser les données standardisées pour SVM et Logistic Regression
        if name in ['SVM', 'Logistic Regression']:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # Entraînement
        model.fit(X_tr, y_train)
        
        # Prédictions
        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='f1_weighted')
        
        # Stockage des résultats
        model_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        trained_models[name] = model
        
        print(f"      ✅ Accuracy: {accuracy:.4f}")
        print(f"      ✅ F1-Score: {f1:.4f}")
        print(f"      ✅ CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return model_results, trained_models, scaler_clf
