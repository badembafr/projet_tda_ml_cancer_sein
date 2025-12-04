# Projet Cancer - Classification du Cancer du Sein

**Auteur:** Bademba SANGARE (22009816)  
**Formation:** Master 1 Informatique et Big Data - Groupe 2  
**Année:** 2025-2026  
**Professeur:** Rakia JAZIRI  **Cours:** Techniques d'apprentissage artificiel

**Dépôt GitHub:** https://github.com/badembafr/projet_tda_ml_cancer_sein

## Problématique

Comment prédire avec précision le diagnostic du cancer du sein (bénin ou malin) à partir des caractéristiques des cellules tumorales ?

## Description

Ce projet compare trois algorithmes de machine learning (Random Forest, SVM, KNN) pour classifier des tumeurs du sein.

Etapes: chargement des données, normalisation, division train/test, entrainement des modèles, évaluation, comparaison des performances et visualisation des résultats.

## Données

Dataset: 569 échantillons avec 32 caractéristiques de tumeurs (357 bénignes, 212 malignes).

## Structure du Projet

```
data/raw/               # breast_cancer_data.csv
models/                 # Modèles sauvegardés
reports/figures/        # Graphiques (ROC, matrices de confusion)
src/                    # Code source (dataset, modeling, plots)
train.py                # Script principal
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Entraîner les modèles

```bash
python3 train.py
```

Le script charge les données (569 échantillons), entraîne Random Forest, SVM et KNN, compare leurs performances et sauvegarde le meilleur modèle avec les graphiques dans `reports/figures/`.

## Résultats

Random Forest et SVM atteignent 97.37% d'accuracy, KNN 95.61%. Random Forest est sélectionné comme meilleur modèle avec une précision de 100% et un recall de 92.86% sur les tumeurs malignes.

Images de représentation générées: courbes ROC et matrices de confusion dans `reports/figures/`.

## Configuration

Paramètres dans `src/config.py` :
- `RANDOM_STATE` : Seed pour reproductibilité
- `TEST_SIZE` : Proportion du jeu de test
- `N_ESTIMATORS_RF` : Nombre d'arbres pour Random Forest
- `KNN_NEIGHBORS` : Nombre de voisins pour KNN