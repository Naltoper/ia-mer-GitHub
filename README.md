# IA mer
## Meilleur score :
- model : 
```python
SVC = {
    "name" : "SVC",
    "hyper_param" : {
        "dual": False,
        "max_iter": 5000,  
        "C": 0.0001 
    }
}
```

- X : 
```python
['X_grad', 'X_histoHSV', 'X_histo']
```

- Params de resize :
```python
resize_h, resize_l = 128, 128
resize_grad = 128
```

- Resultat :
```bash
- err empirique : 12.56%
- err reel      : 18.83%
```



Nombre d'image dans le sample S :  808
Entrainement terminé avec X = ['X_grad', 'X_histoHSV', 'X_histo']
Model : KNN
err empirique : 17.33%
err reel      : 20.42%

Nombre d'image dans le sample S :  808
Entrainement terminé avec X = ['X_grad', 'X_histoHSV', 'X_histo']
Model : SVC
err empirique : 14.36%
err reel      : 19.18%

Nombre d'image dans le sample S :  808
Entrainement terminé avec X = ['X_grad', 'X_histoHSV', 'X_histo']
Model : RFC
err empirique : 11.14%
err reel      : 20.92%