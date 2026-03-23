import joblib
from src.preprocessing import *
from src.models import *
from src.metrics import *
from src.votePredict import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os

# 

NAIVE = {
        "name" : "NaiveBayes",
        "hyper_param" : {}
    }

# Improved KNN setup
KNN = {
    "name": "KNN",
    "hyper_param": {
        'n_neighbors': 7,            # smaller value works better for image embeddings
        'weights': 'distance',       # closer neighbors count more
        'metric': 'cosine'           # cosine distance often works well for high-dimensional image features
    }
}
param_grid_KNN = {
    'n_neighbors': [5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

LINEAR_SVC = {
        "name" : "LinearSVC",
        "hyper_param" : {
            'C': 0.0001, 
            'dual': False, 
            'max_iter': 5000, 
            'penalty': 'l2'
            }
    }
param_grid_LINEAR_SVC = {
    'C': [0.00001, 0.0001, 0.001, 0.1], # contrôle la force de la régularisation
    'dual': [False], 
    'penalty': ['l2'],
    'max_iter': [5000, 10000]
}
   
RFC = {
    "name": "RFC",
    "hyper_param": {
        'n_estimators': 200,          # more trees
        'max_depth': None,            # allow deep trees
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',       # features considered at each split
        'random_state': 42
    }
}
param_grid_RFC = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
}
  
GB = {
    "name": "GB",
    "hyper_param": {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42
    }
}
param_grid_GB = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
}
   
   
def hyper_param_research(features_to_use: list[str], S: list[dict], model,param_grid: dict):
    # Avant de lancer cette fonction il faut :
    # définir la grille des paramètres à tester
    # initialiser le modèle de base

    # On configure la recherche (cv=5 signifie qu'il divise le set d'images en 5 morceaux)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                            cv=5, scoring='accuracy', n_jobs=-1, verbose=1,
                            return_train_score=True)
    
    # On extrait X et y
    X_train = extract_X(features_to_use, S)
    y_train = np.array([img['y_true_class'] for img in S])

    # On fit sur (X_train, y_train)
    grid_search.fit(X_train, y_train)

    # On récupère les meilleurs paramètres
    results = grid_search.cv_results_

    best_idx = -1
    best_score_robust = -1
    
    # ON VEUT LES PARAM OPTIMAUX, PAS JUSTE LES MEILLEURS
    # On parcourt tous les tests effectués par GridSearchCV
    for i in range(len(results['params'])):
        mean_test = results['mean_test_score'][i]
        mean_train = results['mean_train_score'][i]
        gap = mean_train - mean_test
        
        # CRITÈRE D'OPTIMALITÉ : 
        # On veut le meilleur score de test, 
        # MAIS on ignore les modèles où l'overfit est trop grand (> 10%)
        if gap < 0.10: 
            if mean_test > best_score_robust:
                best_score_robust = mean_test
                best_idx = i

    # Si aucun modèle n'est sous les 10% d'écart, on prend quand même le meilleur par défaut
    if best_idx == -1:
        best_idx = grid_search.best_index_

    optimal_params = results['params'][best_idx]
    
    print("\n--- Paramètres Optimaux (Robustes) ---")
    print(f"Paramètres : {optimal_params}")
    print(f"Validation Score : {results['mean_test_score'][best_idx]:.2%}")
    print(f"Overfit Gap : {(results['mean_train_score'][best_idx] - results['mean_test_score'][best_idx]):.2%}")
    
    return optimal_params   
    
    


