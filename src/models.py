import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Ici les fonctions d'entrainement, predictions des models
scaler = StandardScaler()

def extract_X(features_to_use: list[str], S: list[dict], model, training: bool):
    """
    Créer X en fonction des features choisies.
    Features possibles: ['X_histo', X_histoHSV, 'X_grad']
    Retourne un nparray scaled par un Scaler enregistré comme un attribut dans l'objet model
    """
    data_list = []
    for img in S:
        combined_features = []
        
        for feat in features_to_use:
            if feat in ['X_histo', 'X_histoHSV', 'X_grad']:
                # Pour X_histo ou d'autres features directes (listes ou arrays) rien à faire
                array = img[feat]
                combined_features.extend(array)
            else:
                raise ValueError(f"Unsupported feature: {feat}")
                    
        data_list.append(combined_features)
        
    X = np.array(data_list)
    
    if training:
        # Phase d'entraînement : on crée le scaler, on le sauvegarde et on l'apprend (fit)
        model.scaler = StandardScaler()
        X = model.scaler.fit_transform(X)
    else:
        # Phase de prédiction : on utilise le scaler déjà stocké
        if hasattr(model, 'scaler') and model.scaler is not None:
            X = model.scaler.transform(X)
        else:
            # si pas de scaler, on lever une erreur
            raise ValueError("No scaler found in model, if you test a model, it must have been trained with this fonction, with training=True")
    
    return X


def chooseAlgo(algo):
    if algo['name'] == "NaiveBayes":
        model = GaussianNB(**algo.get('hyper_param', {}))
    elif algo['name'] == "KNN":
        model = KNeighborsClassifier(**algo.get('hyper_param', {}))
    elif algo['name'] == "LinearSVC":
        model = LinearSVC(**algo.get('hyper_param', {})) 
    elif algo['name'] == "RFC":
        model = RandomForestClassifier(**algo.get('hyper_param', {})) 
    else:
        raise ValueError(f"Unsupported algorithm: {algo['name']}")
    
    return model



def fitFrom(features_to_use: list[str], S_train: list, algo: dict):
    """
    Entraîne un modèle en utilisant les features données (features_to_use).
    """
    model = chooseAlgo(algo)
    # On creer un attribut features pour les retrouver plus tard
    model.features = features_to_use
    
    # Caracteristique et cible
    X_train = extract_X(features_to_use, S_train, model, training=True) # training=True : on fit et save un scaler dans ce model
    y_train = np.array([img['y_true_class'] for img in S_train])
    
    
    
    # Entraînement
    model.fit(X_train, y_train)
    
    print(f"Entrainement terminé avec X = {features_to_use}")
    print(f"Model : {algo['name']}")
    return model


def predictFrom(S_test: list[dict], model):
    """
    Calcule la prédiction du modèle sur les données de S_test (sur les memes features que son entrainement).
    Modifie S_test en place en ajoutant/mettant à jour 'y_predicted_class'.
    """
    X_test = extract_X(model.features , S_test, model, training=False) # Fase : extract_X utilisera le scaler saved quand le model a ete entrainé
    
    predictions = model.predict(X_test)
    
    # Modif de S
    # Zip pour parcourir simultanément les dictionnaires et les prédictions
    for img, pred in zip(S_test, predictions):
        img['y_predicted_class'] = int(pred)
    
    return S_test




if __name__ == '__main__':
    pass
    