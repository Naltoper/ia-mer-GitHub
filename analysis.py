import os
from PIL import Image
import cv2
import numpy as np

# Ici les fonctions de manipulations, traitement des données

path_mer = "../Data/Mer"
path_Ailleurs = "../Data/Ailleurs"
image_test = '../Data/Mer/js44.jpeg'
resize_h, resize_l = 128, 128
resize_grad = 128


def resizeImage(imgPath: str, h: int, l: int) -> Image:
    """
    retourne l image retaillée (et la sauvegarde dans ../Data/Resized/) pour qu elle soit de taille h*l
    """
    img = Image.open(imgPath).convert('RGB')
    img_resize = img.resize((h,l))
    #name = os.path.splitext(os.path.basename(imgPath))[0] + '_resized.jpg'
    #img_resize.save(f"../Data/Resized/{name}")
    return img_resize


def computeHisto(image: Image.Image):
    """
    calcule et retourne (et/ou stocke) l histogramme de couleurs 
    de l image i, à l aide de la librairie PIL
    return le nombre de pixel de chaque niveau de rouge, vert, bleu
    """
    image.convert('RGB')
    histo = image.histogram()
    return histo


def computeHistoHSV(image: Image.Image):
    """
    calcule et retourne l histogramme de couleurs 
    de l image i en HSV, à l aide de la librairie PIL
    return le nombre de pixel de chaque niveau de rouge, vert, bleu
    """
    imageHSV = image.convert('HSV')
    histo = imageHSV.histogram()
    return histo


def buildSampleFromPath (path1 :str,  path2 :str):
    """
    retourne  un  échantillon  S (list de dict)  de 
    données issu des répertoires path1 (les données +1, avec paysage maritime) et 
    path2 (les données -1, sans paysage maritime)
    """
    
    data_res = []

    #### data ailleurs ####
    data_ailleurs = os.listdir(path1)
    
    for img in data_ailleurs:
        
        name_path = f"{path1}/{img}"
        image128 = resizeImage(name_path, resize_h, resize_l)
        image64 = resizeImage(name_path, resize_grad, resize_grad) # pour ne pas avoir des matrices trop grandes
        
        img = { "name_path" : name_path,
                "resized_image" : image128,
                "X_histo" : computeHisto(image128),
                "X_grad" : computeGradients(image64),
                "X_histoHSV" : computeHistoHSV(image128),
                "y_true_class" : -1,
                "y_predicted_class" : None
                }
        data_res.append(img)
    
    
    #### data mer ####
    data_mer = os.listdir(path2)
    
    for img in data_mer:
        
        name_path = f"{path2}/{img}"
        image128 = resizeImage(name_path, resize_h, resize_l)
        image64 = resizeImage(name_path, resize_grad, resize_grad) # pour ne pas avoir des matrices trop grandes
        
        img = { "name_path" : name_path,
                "resized_image" : image128,
                "X_histo" : computeHisto(image128),
                "X_grad" : computeGradients(image64),
                "X_histoHSV" : computeHistoHSV(image128),
                "y_true_class" : 1,
                "y_predicted_class" : None
                }
        data_res.append(img)
    
    return data_res


def computeGradients(pil_image: Image.Image):
    """
    Calcule un histogramme des directions des gradients pondéré par la magnitude.
    """
    img_gray = pil_image.convert('L')
    img_array = np.array(img_gray)

    # Calcul des gradients X et Y
    grad_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)

    # Conversion en polaires (0 à 360 degrés)
    magnitude, direction = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

    # Création de l'histogramme de direction (8 secteurs de 45°)
    # On utilise np.histogram sur la matrice direction
    # weights=magnitude permet de donner plus d'importance aux contours nets
    histo_direction, _ = np.histogram(direction, bins=8, range=(0, 360), weights=magnitude)

    # Normalisation pour que la taille de l'image n'influence pas les valeurs
    if np.sum(histo_direction) > 0:
        histo_direction = histo_direction / np.sum(histo_direction)

    # On peut garder quelques stats globales en plus
    gradStats = [
        np.mean(magnitude) / 255.0, # Normalisé
        np.std(magnitude) / 255.0
    ]
    
    # On combine les stats et l'histogramme (8 valeurs)
    return gradStats + histo_direction.tolist()





if __name__ == '__main__':
    # S = buildSampleFromPath(path_Ailleurs, path_mer)
    # print(S[0])
    
    # image64 = resizeImage('../Data/Mer/ra7jj.jpeg', 64, 64)
    # magnetude, direction = computeGradients(image64)
    # print(len(magnetude))
    # # Pour voir les 5x5 premiers pixels de la matrice
    # print("Extrait de la magnitude :\n", magnetude[:5, :5])
    
    # img = resizeImage('../Data/Mer/qhsrty65.jpg', 4,4)
    # print(computeHistoHSV(img))
    pass