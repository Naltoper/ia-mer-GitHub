from analysis import *
from models import *
from metrics import *
from votePredict import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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

# KNN = {
#         "name" : "KNN",
#         "hyper_param" : {
#             'metric': 'manhattan', 
#             'n_neighbors':25, 
#             'weights': 'uniform'
#             }
#     }
# param_grid_KNN = {
#     'n_neighbors': [25, 31, 41, 51], 
#     'weights': ['uniform'], 
#     'metric': ['euclidean', 'manhattan']
# }
 
SVC = {
        "name" : "SVC",
        "hyper_param" : {
            'C': 0.0001, 
            'dual': False, 
            'max_iter': 5000, 
            'penalty': 'l2'
            }
    }
param_grid_SVC = {
    'C': [0.00001, 0.0001, 0.001, 0.1], # contrôle la force de la régularisation
    'dual': [False], 
    'penalty': ['l2'],
    'max_iter': [5000, 10000]
}
   
RFC = {
    "name": "RFC",
    "hyper_param": {
        'max_depth': 10, 
        'max_features': 'log2', 
        'min_samples_leaf': 10, 
        'min_samples_split': 2, 
        'n_estimators': 50
        }
}
param_grid_RFC = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2']
    }    
   
   
def hyper_param_research(features_to_use: list[str], S: list[dict], model,param_grid: dict):
    # On définit la grille des paramètres à tester

    # On initialise le modèle de base

    # On configure la recherche (cv=5 signifie qu'il divise tes 414 images en 5 morceaux)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                            cv=5, scoring='accuracy', n_jobs=-1, verbose=1,
                            return_train_score=True)
    
    # On extrait X et y
    X_train = extract_X(features_to_use, S)
    y_train = np.array([img['y_true_class'] for img in S])

    # On lance l'entraînement sur tes données (X_train, y_train)
    grid_search.fit(X_train, y_train)

    # On récupère les meilleurs paramètres
    results = grid_search.cv_results_

    best_idx = -1
    best_score_robust = -1
    
    # ON VEUT LES PARAM OPTIMAUX, PAS LES MEILLEURS
    # On parcourt tous les tests effectués par GridSearchCV
    for i in range(len(results['params'])):
        mean_test = results['mean_test_score'][i]
        mean_train = results['mean_train_score'][i]
        gap = mean_train - mean_test
        
        # CRITÈRE D'OPTIMALITÉ : 
        # On veut le meilleur score de test, 
        # MAIS on ignore les modèles où l'overfit est trop violent (> 10%)
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
    
    
# Sample complet
S = buildSampleFromPath(path_Ailleurs, path_mer)
print("Nombre d'image dans le sample S : ", len(S))
features = ['X_grad', 'X_histoHSV', 'X_histo']

# hyper_param_research(features, S, RandomForestClassifier(), param_grid_RFC)


# model1 = fitFrom(['X_grad', 'X_histoHSV'], S, NAIVE)
model2 = fitFrom(['X_grad', 'X_histoHSV', 'X_histo'], S, KNN)
model3 = fitFrom(['X_grad', 'X_histoHSV', 'X_histo'], S, SVC)
model4 = fitFrom(['X_grad', 'X_histoHSV', 'X_histo'], S, RFC)

print(f"err empirique : {err_empirique_vote(S, model2, model3, model4):.2%}")
print(f"err reel      : {err_real_cv_vote(S, KNN, SVC, RFC, features):.2%}")


# print(f"err empirique : {err_empirique(S, model4):.2%}")
# print(f"err reel      : {err_real_cv(S, model4):.2%}")









