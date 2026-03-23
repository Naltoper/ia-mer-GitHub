from src.config import *


# Sample complet
S = buildSampleFromPath(path_Ailleurs, path_mer)
print("Nombre d'image dans le sample S : ", len(S))
features = ['X_grad', 'X_histoHSV', 'X_histo']

######## RECHERCHE HYPER-PARAMS ###########################################################
# hyper_param_research(features, S, RandomForestClassifier(), param_grid_RFC)

######## ENTRAINEMENT ###########################################################

# model_naive = fitFrom(['X_grad', 'X_histoHSV'], S, NAIVE)
model_knn = fitFrom(features, S, KNN)
model_linearSVC = fitFrom(features, S, LINEAR_SVC)
model_rfc = fitFrom(features, S, RFC)


######## METRICS DES MODELS ###################################################
# SOLO
# print(f"err empirique : {err_empirique(S, model_rfc):.2%}")
# print(f"err reel      : {err_real_cv(S, model_rfc):.2%}")

# ENSEMBLE
# print(f"err empirique : {err_empirique_vote(S, model_knn, model_linearSVC, model_rfc):.2%}")
# print(f"err reel      : {err_real_cv_vote(S, KNN, LINEAR_SVC, RFC, features):.2%}")

