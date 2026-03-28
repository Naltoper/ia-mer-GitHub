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
print(f"err empirique : {err_empirique_vote5(S, model_knn, model_linearSVC, model_rfc, model_gb, model_SVC):.2%}")
print(f"err reel      : {err_real_cv_vote5(S, KNN, LINEAR_SVC, RFC, GB, SVC_, features):.2%}")

