import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler





def optimize_best_model(best_model_name, trained_models, X_train, y_train):
    """
    MODULE 9: Optimise les hyperparamètres du meilleur modèle
    
    Cette fonction effectue une recherche d'hyperparamètres pour améliorer
    les performances du meilleur modèle identifié.
    """
    print(f"\n🔧 MODULE 9: OPTIMISATION DES HYPERPARAMÈTRES - {best_model_name}")
    print("-" * 60)
    
    # Définition des grilles de paramètres
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'criterion': ['gini', 'entropy']
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs']
        }
    }
    
    if best_model_name not in param_grids:
        print(f"❌ Grille de paramètres non définie pour {best_model_name}")
        return trained_models[best_model_name]
    
    print(f"🔍 Recherche des meilleurs hyperparamètres pour {best_model_name}...")
    
    # GridSearchCV
    base_model = trained_models[best_model_name]
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grids[best_model_name],
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # Préparation des données selon le modèle
    if best_model_name in ['SVM', 'Logistic Regression']:
        scaler_opt = StandardScaler()
        X_train_opt = scaler_opt.fit_transform(X_train)
    else:
        X_train_opt = X_train
        scaler_opt = None
    
    grid_search.fit(X_train_opt, y_train)
    
    print(f"✅ Optimisation terminée")
    print(f"🎯 Meilleurs paramètres: {grid_search.best_params_}")
    print(f"📊 Meilleur score F1 (CV): {grid_search.best_score_:.4f}")
    
    # Sauvegarde du modèle optimisé
    optimized_model_data = {
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'scaler': scaler_opt
    }
    
    joblib.dump(optimized_model_data, f'../models/optimized_{best_model_name.lower().replace(" ", "_")}_model.pkl')
    print(f"💾 Modèle optimisé sauvegardé dans models/")
    
    return grid_search.best_estimator_, scaler_opt
