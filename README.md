# Api_Categorize
Api_Catego est une api qui permet de faire l'entraînement d'un modèle de la  catégorisation des transactions bancaires.
L’objectif est de prévoir la catégorie d’une transaction à partir de son descriptif en entraînant un pipeline qui inclut directement toutes les étapes, du nettoyage  de texte jusqu'à l’apprentissage du modèle.
Api_Catego commence par  entraîner deux modèle RandomForest et LogisticRegression, puis la sélection de la bonne combinaison des hyperparamètres au sens de la validation croisé k-folds, au final elle fournit le meilleure modèle qu’il sera en suite mis en production 
Tous cela est bien automatisé,vous allez qu'à fournir le  jeux de données d'entraînement pour entraîner le modèle ou celui  pour faire des prédictions .

## Installation
Les étapes d'installation sont les suivantes
```bash
git clone ...... 
cd catego
docker build -t nom_image . 
```
#Démarrage
