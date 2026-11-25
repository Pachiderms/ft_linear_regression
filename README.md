# ft_linear_regression

Introduction au machine learning : prédire le prix d'une voiture à partir de son
kilométrage grâce à une régression linéaire entraînée par descente de gradient.

Hypothèse : `estimatePrice(km) = theta0 + theta1 * km`

## Installation

Python 3.13 et [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
```

## Utilisation

Les scripts se lancent depuis la racine du dépôt:

### sans le venv et sans modules

```bash
# 1. Entraîner le modèle (écrit thetas.json)
uv run src/train.py                              # utilise data.csv
uv run src/train.py data_km_price_expanded.csv   # autre dataset

# 2. Prédire un prix
uv run src/predict.py
# Kilométrage : 100000
# Prix estimé : 6354.70
```

### avec le venv

```bash
# 1. Activer l'environnement virtuel
source .venv/bin/activate

# 2. Entraîner le modèle (écrit thetas.json)
python src/train.py                              # utilise data.csv
python src/train.py data_km_price_expanded.csv   # autre dataset

# 3. Prédire un prix
python src/predict.py
# Kilométrage : 100000
# Prix estimé : 6354.70
```

### avec les modules + le venv

```bash
# 1. Entraîner le modèle (écrit thetas.json)
python -m src.train                              # utilise data.csv
python -m src.train data_km_price_expanded.csv   # autre dataset

# 2. Prédire un prix
python -m src.predict
# Kilométrage : 100000
# Prix estimé : 6354.70
```

### avec les modules sans le venv

```bash
# 1. Entraîner le modèle (écrit thetas.json)
uv run python -m src.train                              # utilise data.csv
uv run python -m src.train data_km_price_expanded.csv   # autre dataset

# 2. Prédire un prix
uv run python -m src.predict
# Kilométrage : 100000
# Prix estimé : 6354.70 avec un modèle train sur data.csv
```

Sans `thetas.json` (modèle non entraîné), `predict` renvoie 0, car theta0 et theta1
valent 0 au départ.

### Bonus

```bash
uv run python -m src.plot               # nuage de points + droite
uv run python -m src.plot --save        # enregistre regression.png
uv run python -m src.precision          # MAE, RMSE, R2
```

## Fonctionnement

- `src/main.py` : `MyLinearRegression` (hypothèse, gradient, descente de gradient).
  theta0 et theta1 démarrent à 0 et sont mis à jour simultanément :
  `theta := theta - alpha * (1/m) * Xᵀ (X·theta - y)`.
- `src/train.py` : normalise les kilométrages (z-score) pour que la descente de
  gradient converge, entraîne le modèle, puis **remet theta0 et theta1 à l'échelle
  d'origine** avant de les sauvegarder dans `thetas.json`. Ainsi `predict.py` applique
  directement `theta0 + theta1 * km`, sans avoir besoin de la moyenne ni de l'écart-type.
- `src/loss.py` : fonctions de coût et de métriques (MSE, RMSE, MAE, R²).

## Résultats (`data.csv`)

| theta0  | theta1   | MAE    | RMSE   | R²    |
| ------- | -------- | ------ | ------ | ----- |
| 8499.60 | -0.02145 | 557.84 | 667.57 | 0.733 |

## Utilisation de l'IA

- génération de données d'entrainement supplémentaire
- révision de code (exemple: inversion du y et du y*hat pour le r2score*)

## Notes

Aucune bibliothèque ne fait la régression utilisée : numpy sert au calcul matriciel,
pandas à la lecture du CSV et matplotlib aux graphiques.
