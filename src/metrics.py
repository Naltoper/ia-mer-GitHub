import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from src.models import extract_X, fitFrom, predictFrom
from src.votePredict import votePredict3

# Ici les fonction d'evaluations des performances des models

def err_empirique(S_train: list[dict], model_trained):
    """
    Calcule l'erreur empirique du modèle.
    Retourne la proportion d'erreur sur une prediction.
    Donc S_predicted est l'echantillon sur lequel
    le model s'est entrainé puis a fait ses predictions.
    """
    nb_erreurs = 0
    total = len(S_train)
    # On fait les predictions avec le sample qui a appris le model
    S_predicted = predictFrom(S_train, model_trained)
    
    # Pour chaque img on compare la prediction a 'y_true_class'
    for img in S_predicted:
        if img['y_true_class'] != img['y_predicted_class']:
            nb_erreurs += 1
            
    return nb_erreurs / total


def err_real_cv(S: list[dict], model_trained):
    """
    Calcule l'erreur réelle par validation croisée sur l'ensemble S.
    """
    # Préparation des données complètes (Conversion en NumPy pour le type-checker)
    X = extract_X(model_trained.features, S, model_trained, training=False)
    y = np.array([img['y_true_class'] for img in S])
    
    # Cross-validation
    # cross_val_score clone et entraîne sur différentes parties de X/y.
    scores = cross_val_score(model_trained, X, y, cv=5)
    
    # Calcul de l'erreur
    errors = 1 - scores
    return errors.mean()
    

def err_empirique_vote(S_train: list[dict], model1, model2, model3):
    """
    Calcule l'erreur empirique du modèle.
    Retourne la proportion d'erreur sur une prediction.
    Donc S_predicted est l'echantillon sur lequel
    le model s'est entrainé puis a fait ses predictions.
    """
    nb_erreurs = 0
    total = len(S_train)
    # On fait les predictions avec le sample qui a appris le model
    S_predicted = votePredict3(S_train, model1, model2, model3)
    
    # Pour chaque img on compare la prediction a 'y_true_class'
    for img in S_predicted:
        if img['y_true_class'] != img['y_predicted_class']:
            nb_erreurs += 1
            
    return nb_erreurs / total


def err_real_cv_vote(S: list[dict], algo1, algo2, algo3, features: list[str], cv=5):
    """
    Simule une validation croisée pour le système de vote.
    'algo1, algo2, algo3' sont les dictionnaires de config (ex: SVC, KNN).
    'features' est la liste des features à utiliser (ex: ['X_histo'])
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    S_np = np.array(S) # Pour faciliter le slicing
    errors = []

    for train_index, test_index in kf.split(S_np):
        # Séparation Train / Test
        S_train = S_np[train_index].tolist()
        S_test = S_np[test_index].tolist()

        # Entraînement des 3 modèles sur S_train
        m1 = fitFrom(features, S_train, algo1)
        m2 = fitFrom(features, S_train, algo2)
        m3 = fitFrom(features, S_train, algo3)

        # Prédiction par vote sur S_test
        S_voted = votePredict3(S_test, m1, m2, m3)

        # Calcul de l'erreur sur ce fold
        nb_err = 0
        for img in S_voted:
            # On compare y_true et y_predicted
            if img['y_true_class'] != img['y_predicted_class']:
                nb_err = nb_err + 1
                
        errors.append(nb_err / len(S_test))

    return np.mean(errors)
