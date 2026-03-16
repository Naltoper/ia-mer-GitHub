from main import *

# Sample complet
S = buildSampleFromPath(path_Ailleurs, path_mer)
print("Nombre d'image dans le sample S : ", len(S))
features = ['X_grad', 'X_histoHSV', 'X_histo']

######## ENTRAINEMENT ###########################################################

# model_naive = fitFrom(['X_grad', 'X_histoHSV'], S, NAIVE)
# model_knn = fitFrom(features, S, KNN)
# model_linearSVC = fitFrom(features, S, LINEAR_SVC)
# model_rfc = fitFrom(features, S, RFC)


# Build S pour Dataset melangé, avec trueclas = None, 
def buildSampleFromPathTEST (path):
    data_res = []

    #### data sans true class ####
    data = os.listdir(path)
    
    for img in data:
        
        name_path = f"{path}/{img}"
        image_resized_histo = resizeImage(name_path, resize_h, resize_l)
        image_resized_grad = resizeImage(name_path, resize_grad, resize_grad) # pour ne pas avoir des matrices trop grandes
        
        img = { "name_path" : name_path,
                "resized_image" : image_resized_histo,
                "X_histo" : computeHisto(image_resized_histo),
                "X_grad" : computeGradients(image_resized_grad),
                "X_histoHSV" : computeHistoHSV(image_resized_histo),
                "y_true_class" : None,
                "y_predicted_class" : None
                }
        data_res.append(img)
    
    return data_res

def save_models(models_dict, filename="models_roblof.joblib"):
    """ Sauvegarde un dictionnaire de modèles dans un fichier """
    joblib.dump(models_dict, filename)
    print(f"Modèles sauvegardés dans {filename}")

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

######## SAUVEGARDE DES MODELS ###################################################
# Une fois les modèles entraînés :
mes_modeles = {
    'knn': model_knn,
    'svc': model_linearSVC,
    'rfc': model_rfc
}
# save_models(mes_modeles)


######## TEST CC2 #############################################################0.22

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