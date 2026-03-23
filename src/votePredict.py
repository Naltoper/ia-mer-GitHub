import copy

from src.models import predictFrom

def votePredict3(S: list[dict], model1, model2, model3) -> list[dict]:
    """
    Effectue une prédiction basée sur le vote majoritaire de 3 modèles.
    Modifie S en place et met à jour 'y_predicted_class'.
    """
    
    # On récupère les prédictions de chaque modèle séparément
    res1 = predictFrom(copy.deepcopy(S), model1)
    res2 = predictFrom(copy.deepcopy(S), model2)
    res3 = predictFrom(copy.deepcopy(S), model3)

    # On compare les résultats pour chaque image
    for i in range(len(S)):
        p1 = res1[i]['y_predicted_class']
        p2 = res2[i]['y_predicted_class']
        p3 = res3[i]['y_predicted_class']
        
        # Vote majoritaire : 
        # Si on a deux +1, la somme sera >= 1
        # Si on a deux -1, la somme sera <= -1
        vote_final = 1 if (p1 + p2 + p3) >= 1 else -1
        
        # On met à jour l'échantillon original
        S[i]['y_predicted_class'] = vote_final

    return S

if __name__ == '__main__':
    pass