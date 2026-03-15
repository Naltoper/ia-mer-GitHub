import joblib

from analysis import *
from models import *
from metrics import *
from votePredict import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import os

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

# RFC = {
#     "name": "RFC",
#     "hyper_param": {
#         'max_depth': 10, 
#         'max_features': 'log2', 
#         'min_samples_leaf': 10, 
#         'min_samples_split': 2, 
#         'n_estimators': 50
#         }
# }
# param_grid_RFC = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 5, 10, 15],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 5, 10],
#         'max_features': ['sqrt', 'log2']
#     }    
   
   
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
    
def generate_cc2_file(S_test, model_knn, model_svc, model_rfc, filename="roblof.txt"):
    """
    Génère le fichier de résultats au format CC2 en utilisant le vote majoritaire.
    """
    # Calcul des prédictions via le vote majoritaire
    # remplie 'y_predicted_class'
    S_test = votePredict3(S_test, model_knn, model_svc, model_rfc)
    
    # footer
    ee = 0.00  
    er = 0.17  

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # header
            f.write("# Florent Fabretti, Vincent Fabretti, Djibril Mimouni (Equipe roblof)\n")
            f.write("# Vote majoritaire (Ensemble : KNN, LinearSVC, RandomForest)\n")
            f.write("# KNN(k=7,cosine), SVC(C=0.0001), RFC(n=200,max_depth=None)\n")
            f.write("# Concaténation (sur image 128x128): Gradients résumé, Histo HSV, Histo RGB\n")
            
            # Liste des images et prédictions
            for img in S_test:
                # On extrait juste "image.jpg" du path
                clean_name = os.path.basename(img['name_path'])
                pred = img['y_predicted_class']
                
                # Formatage : Nom +1 ou Nom -1
                pred_str = f"+{pred}" if pred > 0 else f"{pred}"
                
                f.write(f"{clean_name} {pred_str}\n")
            
            # Footer
            f.write(f"# EE = {ee:.2f}\n")
            f.write(f"# ER = {er:.2f}\n")

        print(f"Fichier {filename} généré avec succès.")
        
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier : {e}")
    
    
# Sample complet
# S = buildSampleFromPath(path_Ailleurs, path_mer)
# print("Nombre d'image dans le sample S : ", len(S))
# features = ['X_grad', 'X_histoHSV', 'X_histo']

# hyper_param_research(features, S, RandomForestClassifier(), param_grid_RFC)

######## ENTRAINEMENT ###########################################################

# model_naive = fitFrom(['X_grad', 'X_histoHSV'], S, NAIVE)
# model_knn = fitFrom(features, S, KNN)
# model_linearSVC = fitFrom(features, S, LINEAR_SVC)
# model_rfc = fitFrom(features, S, RFC)

######## SAUVEGARDE DES MODELS ###################################################

# def save_models(models_dict, filename="models_roblof.joblib"):
#     """ Sauvegarde un dictionnaire de modèles dans un fichier """
#     joblib.dump(models_dict, filename)
#     print(f"Modèles sauvegardés dans {filename}")

# # --- Dans ton bloc principal ---
# # Une fois les modèles entraînés :
# mes_modeles = {
#     'knn': model_knn,
#     'svc': model_linearSVC,
#     'rfc': model_rfc
# }
# save_models(mes_modeles)


######## METRICS DES MODELS ###################################################

# print(f"err empirique : {err_empirique_vote(S, model_knn, model_linearSVC, model_rfc):.2%}")
# print(f"err reel      : {err_real_cv_vote(S, KNN, LINEAR_SVC, RFC, features):.2%}")

# print(f"err empirique : {err_empirique(S, model_rfc):.2%}")
# print(f"err reel      : {err_real_cv(S, model_rfc):.2%}")


######## TEST CC2 #############################################################

# print(f"err empirique : {err_empirique_vote(S, model_knn, model_linearSVC, model_rfc):.2%}")
# print(f"err reel      : {err_real_cv_vote(S, KNN, LINEAR_SVC, RFC, features):.2%}")

# print(f"err empirique : {err_empirique(S, model_rfc):.2%}")
# print(f"err reel      : {err_real_cv(S, model_rfc):.2%}")


######## TEST CC2 #############################################################

def load_models(filename="models_roblof.joblib"):
    """ Charge les modèles depuis le fichier """
    if os.path.exists(filename):
        print(f"Chargement des modèles depuis {filename}...")
        return joblib.load(filename)
    else:
        print("Erreur : Fichier de modèles introuvable !")
        return None


# Au lieu de fitFrom(), on charge directement :
modeles_charges = load_models("models_roblof.joblib")

if modeles_charges:
    model_knn = modeles_charges['knn']
    model_linearSVC = modeles_charges['svc']
    model_rfc = modeles_charges['rfc']

print("Test sur le sample inconnue...")
S_TEST = buildSampleFromPathTEST(path_TEST)

# generer le fichier pour cc2
generate_cc2_file(S_TEST, model_knn, model_linearSVC, model_rfc)
# TODO save le model pour pas savoir a entrainer a chaque fois

