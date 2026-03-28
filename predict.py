from src.config import *

# Build S pour Dataset inconnnu, avec trueclas = None, 
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

def save_models(models_dict, filename="output/models_roblof.joblib"):
    """ Sauvegarde un dictionnaire de modèles dans un fichier """
    joblib.dump(models_dict, filename)
    print(f"Modèles sauvegardés dans {filename}")

def load_models(filename="models_roblof.joblib"):
    """ Charge les modèles depuis le fichier """
    if os.path.exists(filename):
        print(f"Chargement des modèles depuis {filename}...")
        return joblib.load(filename)
    else:
        print("Erreur : Fichier de modèles introuvable !")
        return None

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

def generate_cc2_file_2nd_passage(S_test, filename="output/roblof2.txt"):
    """
    Génère le fichier de résultats au format CC2 en utilisant le vote majoritaire.
    """
    # footer
    ee = 0.00  
    er = 0.1028

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # header
            f.write("# Florent Fabretti, Vincent Fabretti, Djibril Mimouni (Equipe roblof)\n")
            f.write("# Vote majoritaire (Ensemble : KNN, LinearSVC, RandomForest, GradientBoosting, SVC)\n")
            f.write("# KNN(k=7,cosine), LinearSVC(C=0.0001), RFC(n=200,max_depth=None), GB(learning_rate: 0.1), SVC(C: 10)\n")
            f.write("# 1er passage avec concaténation (sur image 128x128): Gradients résumé, Histo HSV, Histo RGB\n")
            f.write("# 2nd passage avec meme features + y_predicted_class du 1er passage\n")
            
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


# # Sample complet
# S = buildSampleFromPath(path_Ailleurs, path_mer)
# print("Nombre d'image dans le sample S : ", len(S))




# ######## ENTRAINEMENT 1er passage ###########################################################
# # Features 1er passage
# features = ['X_grad', 'X_histoHSV', 'X_histo']

# model_knn = fitFrom(features, S, KNN)
# model_linearSVC = fitFrom(features, S, LINEAR_SVC)
# model_rfc = fitFrom(features, S, RFC)
# model_gb = fitFrom(features, S, GB)
# model_SVC = fitFrom(features, S, SVC_)


# ######## SAUVEGARDE DES MODELS de 1er passage ###################################################
# # Une fois les modèles entraînés :
# mes_modeles_1er = {
#     'knn': model_knn,
#     'linear_svc': model_linearSVC,
#     'rfc': model_rfc,
#     'gb' : model_gb,
#     'svc': model_SVC
# }
# save_models(mes_modeles_1er, filename="output/mes_modeles_1er.joblib")



# ######## ENTRAINEMENT 2nd passage ###########################################################
# # On fait le premier passage pour avoir ses predictions
# votePredict5(S, model_knn, model_linearSVC, model_rfc, model_gb, model_SVC)
# # Features 2nd passage on ajoute les predictions du 1er passage
# features_2nd_passage = ['X_grad', 'X_histoHSV', 'X_histo', 'y_predicted_class']

# model_knn_2nd = fitFrom(features_2nd_passage, S, KNN)
# model_linearSVC_2nd = fitFrom(features_2nd_passage, S, LINEAR_SVC)
# model_rfc_2nd = fitFrom(features_2nd_passage, S, RFC)
# model_gb_2nd = fitFrom(features_2nd_passage, S, GB)
# model_SVC_2nd = fitFrom(features_2nd_passage, S, SVC_)



# ######## SAUVEGARDE DES MODELS de 2nd passage ###################################################
# # Une fois les modèles entraînés :
# mes_modeles_2nd = {
#     'knn': model_knn_2nd,
#     'linear_svc': model_linearSVC_2nd,
#     'rfc': model_rfc_2nd,
#     'gb' : model_gb_2nd,
#     'svc': model_SVC_2nd
# }
# save_models(mes_modeles_2nd, filename="output/mes_modeles_2nd.joblib")


######## PREDICTION #############################################################0.22



# Au lieu de fitFrom(), on charge directement :
# Chargement 1er passage
modeles_charges_1er = load_models("output/mes_modeles_1er.joblib")

if modeles_charges_1er:
    model_knn_1er = modeles_charges_1er['knn']
    model_linearSVC_1er = modeles_charges_1er['linear_svc']
    model_rfc_1er = modeles_charges_1er['rfc']
    model_gb_1er = modeles_charges_1er['gb']
    model_SVC_1er = modeles_charges_1er['svc']

# Chargement 2nd passage
modeles_charges_2nd = load_models("output/mes_modeles_2nd.joblib")

if modeles_charges_2nd:
    model_knn_2nd = modeles_charges_2nd['knn']
    model_linearSVC_2nd = modeles_charges_2nd['linear_svc']
    model_rfc_2nd = modeles_charges_2nd['rfc']
    model_gb_2nd = modeles_charges_2nd['gb']
    model_SVC_2nd = modeles_charges_2nd['svc']


######## CHEMIN DES IMAGES TESTS #########
print("Test sur le sample inconnue...")
path_CC3 = "../../Format rendu CC3-20260323/FormatTestCC3/FormatTestCC3"
S_TEST = buildSampleFromPathTEST(path_CC3)

# 1ER PASSAGE
votePredict5(S_TEST, model_knn_1er, model_linearSVC_1er, model_rfc_1er, model_gb_1er, model_SVC_1er)

# 2ND PASSAGE
votePredict5(S_TEST, model_knn_2nd, model_linearSVC_2nd, model_rfc_2nd, model_gb_2nd, model_SVC_2nd)


# generer le fichier pour cc2
# generate_cc2_file(S_TEST, model_knn, model_linearSVC, model_rfc)
generate_cc2_file_2nd_passage(S_TEST)

