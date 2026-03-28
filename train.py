from src.config import *
from sklearn.svm import SVC


# Sample complet
S = buildSampleFromPath(path_Ailleurs, path_mer)
print("Nombre d'image dans le sample S : ", len(S))
features = ['X_grad', 'X_histoHSV', 'X_histo']


######## RECHERCHE HYPER-PARAMS ###########################################################
# hyper_param_research(features, S, SVC(), param_grid_SVC)



######## ENTRAINEMENT ###########################################################

# model_naive = fitFrom(['X_grad', 'X_histoHSV'], S, NAIVE)
model_knn = fitFrom(features, S, KNN)
model_linearSVC = fitFrom(features, S, LINEAR_SVC)
model_rfc = fitFrom(features, S, RFC)
model_gb = fitFrom(features, S, GB)
model_SVC = fitFrom(features, S, SVC_)


######## METRICS DES MODELS ###################################################
# SOLO
# print(f"err empirique : {err_empirique(S, model_gb):.2%}")
# print(f"err reel      : {err_real_cv(S, model_gb):.2%}")

# ENSEMBLE 3
# print(f"err empirique : {err_empirique_vote(S, model_knn, model_linearSVC, model_rfc):.2%}")
# print(f"err reel      : {err_real_cv_vote(S, KNN, LINEAR_SVC, RFC, features):.2%}")

# ENSEMBLE 5
# print(f"err empirique : {err_empirique_vote5(S, model_knn, model_linearSVC, model_rfc, model_gb, model_SVC):.2%}")
# print(f"err reel      : {err_real_cv_vote5(S, KNN, LINEAR_SVC, RFC, GB, SVC_, features):.2%}")


######## MODEL DE 2ND PASSAGE ##################################################

# D'abord on modifie S avec le model de premier passage
votePredict5(S, model_knn, model_linearSVC, model_rfc, model_gb, model_SVC)

# On ajoute la prediction du premier passage dans les features
features_2nd_passage = ['X_grad', 'X_histoHSV', 'X_histo', 'y_predicted_class']

# On entraine un nouveau model ensembliste avec cette features en plus
model_knn_2nd = fitFrom(features_2nd_passage, S, KNN)
model_linearSVC_2nd = fitFrom(features_2nd_passage, S, LINEAR_SVC)
model_rfc_2nd = fitFrom(features_2nd_passage, S, RFC)
model_gb_2nd = fitFrom(features_2nd_passage, S, GB)
model_SVC_2nd = fitFrom(features_2nd_passage, S, SVC_)

# Le model de 2nd passage est entrainé, on peut l'evaluer :
# ENSEMBLE 5
print(f"err empirique : {err_empirique_vote5(S, model_knn_2nd, model_linearSVC_2nd, model_rfc_2nd, model_gb_2nd, model_SVC_2nd):.2%}")
print(f"err reel      : {err_real_cv_vote5(S, KNN, LINEAR_SVC, RFC, GB, SVC_, features_2nd_passage):.2%}")

