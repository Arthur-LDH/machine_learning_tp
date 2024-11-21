#!/usr/bin/env python3

## Fonctions de preprocessing
# Auteurs : Agence dataservices
# Date : 16/12/2019


import os
import re
import json
import logging
import functools
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing._function_transformer import FunctionTransformer

logger = logging.getLogger(__name__)


class AutoLogTransform(BaseEstimator):
    """Application automatiquement une log transformation sur des données numériques si
    la distribution des variables est asymétriques (abs(skew)>min_skewness) et qu'il y a une amplitude
    supérieure à min_aplitude entre le 10e et le 90e percentile

    WARNING : ATTENTION, VOS DONNEES DOIVENT ETRE STRICTEMENT POSITIVES POUR ASSURER UN BON FONCTIONNEMENT

    Parameters
    ----------
    min_skewness : Float : valeur absolu de l'asymétrie (skewness) requise pour appliquer une log transformation
    min_amplitude : float : valeur minimale de l'amplitude entre le 10e pourcentile et le 90e pourcentile
    requise pour appliquer une log transformation
    """
    def __init__(self, min_skewness=2, min_amplitude=10E3):
        # Set attributes
        self.min_skewness = min_skewness
        self.min_amplitude = min_amplitude

        # Columns on which to apply the transformation
        # Set on fit
        # Attention : sklearn does not support columns name, so we can only use indexes
        # Hence, X input must expose same columns order (this won't be checked)
        self.applicable_columns_index = None
        self.n_cols = None

    def _validate_input(self, X):
        '''Function to validate input format

        Args:
            X: element to validate
        Returns:
            pd.DataFrame: X
        '''
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be a DataFrame or a numpy array")
        if self.n_cols is not None and X.shape[1] != self.n_cols:
            raise ValueError(f"Bad shape ({X.shape[1]} != {self.n_cols})")

        # Copy obligatoire pour ne pas modifier l'original !
        if isinstance(X, pd.DataFrame):
            return X.copy(deep=True)
        else:
            return X.copy()

    def fit(self, X, y=None):
        """Fit transformer

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        Returns
        -------
        self
        """
        X = self._validate_input(X)
        # Si X np array, on transforme en dataframe
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        # Sinon, on reset les noms de colonnes car sklearn ne les gère pas
        else:
            X = X.rename(columns={col: i for i, col in enumerate(X.columns)})
        self.n_cols = X.shape[1]

        # Get applicable columns
        skew = X.skew()
        candidates = list(skew[abs(skew)>self.min_skewness].index)
        if len(candidates) > 0:
            q10 = X.iloc[:, candidates].quantile(q=0.1)
            q90 = X.iloc[:, candidates].quantile(q=0.9)
            amp = q90 - q10
            # Update applicable_columns_index
            self.applicable_columns_index = list(amp[amp>self.min_amplitude].index)

        self.fitted_ = True
        return self

    def transform(self, X):
        """Transform X - apply log on applicable columns
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        Raises
        ------
        ValueError: s'il manque des colonnes
        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        # Validate input
        check_is_fitted(self, 'fitted_')
        X = self._validate_input(X)

        # Si X np array, on transforme en dataframe (on pourrait tout faire en np ?)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # On log transforme les colonnes concernées
        if len(self.applicable_columns_index) > 0:
            X.iloc[:, self.applicable_columns_index] = np.log(X.iloc[:, self.applicable_columns_index])

        # Compatibilité -> on retourne des np array
        return X.to_numpy()

    def fit_transform(self, X, y=None):
        """Apply both fit & transform"""
        self.fit(X)
        return self.transform(X)


class ThresholdingTransform(BaseEstimator):
    """Applique un seuillage aux colonnes fournies. Si des valeurs min et max sont données, le seuillage
    est manuel; sinon il est statistique.
    Parameters
    ----------
    tresholds : list<tuple> : chaque tuple contient (nom_colonne,val_min,val_max) si val_min et/ou val_max
    ne sont pas fournies, le seuillage s'effectue sur les valeurs des quantiles observées
    quantiles : tuple(min_q, max_q)
    """
    def __init__(self, thresholds: list = None, quantiles : tuple = (0.05, 0.95)):
        if thresholds is None:
            raise ValueError("Tresholds is empty, a list<tuple> is required with each tuple : ([val_min], [val_max])")
        if type(quantiles) is not tuple or not 0 < quantiles[0] < 1 or not 0 < quantiles[1] < 1 or not quantiles[0] < quantiles[1]:
            raise ValueError(f"quantiles must be a tuple (quantile_min,quantile_max), default : (0.05, 0.95) \
                with quantile_min < quantile_max and both >0 and <1. quantiles = {quantiles} is not supported.")

        # Set attributes
        self.thresholds = thresholds
        self.fitted_thresholds = []
        self.quantiles = quantiles

    def _validate_input(self, X):
        '''Function to validate input format

        Args:
            X: element to validate
        Returns:
            pd.DataFrame: X
        '''
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be a DataFrame or a numpy array")
        if X.shape[1] != len(self.thresholds):
            raise ValueError(f"Bad shape ({X.shape[1]} != {len(self.thresholds)})")

        # Copy obligatoire pour ne pas modifier l'original !
        if isinstance(X, pd.DataFrame):
            return X.copy(deep=True)
        else:
            return X.copy()

    def fit(self, X, y=None):
        """Fit the ThresholdingTransform on X.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : ThresholdingTransform
        """
        X = self._validate_input(X)
        # Si X np array, on transforme en dataframe
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # On fit chaque colonne 1 par 1
        for col_index, item in enumerate(self.thresholds):
            val_min, val_max = item
            if val_min is None:
                val_min = X.iloc[:, col_index].quantile(q=self.quantiles[0])
            if val_max is None:
                val_max = X.iloc[:, col_index].quantile(q=self.quantiles[1])
            self.fitted_thresholds.append((col_index, val_min, val_max))

        self.fitted_ = True
        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """
        check_is_fitted(self, 'fitted_')
        X = self._validate_input(X)
        # Si X np array, on transforme en dataframe
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for item in self.fitted_thresholds:
            col_index, val_min, val_max = item
            X.iloc[:, col_index][X.iloc[:, col_index] < val_min] = val_min
            X.iloc[:, col_index][X.iloc[:, col_index] > val_max] = val_max

        return X.to_numpy() # Compatibilité -> on retourne des np array

    def fit_transform(self, X, y=None):
        """Apply both fit & transform"""
        self.fit(X)
        return self.transform(X)


class AutoBinner(BaseEstimator):
    """Crée automatiquement une catégorie "other" lorsque les
    cardinalités des catégories sont fortement déséquilibrées
    /!\ Remplace les valeurs de certaines catégories
    Parameters
    ----------
    strategy : 'auto' or 'threshold'

        - 'auto' : On agrège toutes les catégories tant que leur fréquence cumulée est inférieure à threshold
        - 'threshold' : On agrège toutes les catégories dont la fréquence est inférieure à threshold

    min_cat_count : int -> minimum de catégories à garder
    threshold : float
    """
    def __init__(self, strategy="auto", min_cat_count=3, threshold=0.05):
        allowed_strategies = ["threshold", "auto"]
        self.strategy = strategy
        if self.strategy not in allowed_strategies:
            raise ValueError(f"Can only use these strategies: {allowed_strategies}. " +\
                             f"Got strategy={strategy}")
        if not isinstance(min_cat_count, int) or min_cat_count < 0:
            raise ValueError("min_cat_count must be an int and > 0")
        if not isinstance(threshold, float) or 0 < threshold < 0:
            raise ValueError("threshold must be a float in [0,1]")

        # Set attributes
        self.min_cat_count = min_cat_count
        self.threshold = threshold
        self.kept_cat_by_index = {}
        self.n_features = None

    def _validate_input(self, X):
        '''Function to validate input format

        Args:
            X: element to validate
        Returns:
            pd.DataFrame: X
        '''
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be a DataFrame or a numpy array")
        if self.n_features is not None and X.shape[1] != self.n_features:
            raise ValueError(f"Bad shape ({X.shape[1]} != {self.n_features})")

        # Copy obligatoire pour ne pas modifier l'original !
        if isinstance(X, pd.DataFrame):
            return X.copy(deep=True)
        else:
            return X.copy()

    def _set_categories(self, col_index, values):
        self.kept_cat_by_index[col_index] = values

    def fit(self, X, y=None):
        """Fit the AutoBinner on X.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : AutoBinner
        """
        X = self._validate_input(X)
        # Si X np array, on transforme en dataframe
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.n_features = X.shape[1]
        # On analyse colonne par colonne
        for col_index in range(self.n_features):
            # Get col serie
            X_tmp_ser = X.iloc[:, col_index]
            # Get unique vals
            unique_cat = list(X_tmp_ser.unique())
            # If less vals than min threshold, set this column allowed values with all uniques values
            if len(unique_cat) <= self.min_cat_count:
                self._set_categories(col_index, unique_cat)
                continue

            # If more vals than min threshold, keep values based on strategy
            table = X_tmp_ser.value_counts() / X_tmp_ser.count()
            table = table.sort_values()
            if self.strategy == 'auto':
                table = np.cumsum(table)
            # Si une seule cat < threshold -> ça ne sert à rien de la transformer
            if table[1] > self.threshold:
                self._set_categories(col_index, unique_cat)
                continue
            # Sinon, on enlève les catégories en trop
            else:
                to_remove = list(table[table<self.threshold].index)
                for item in to_remove:
                    unique_cat.remove(item)
                self._set_categories(col_index, unique_cat)

        self.fitted_ = True
        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """
        check_is_fitted(self, 'fitted_')
        X = self._validate_input(X)
        # Si X np array, on transforme en dataframe
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for col_index in range(self.n_features):
            X.iloc[:, col_index] = X.iloc[:, col_index].apply(lambda x: x if x in self.kept_cat_by_index[col_index] else 'other_')

        return X.to_numpy() # Compatibilité -> on retourne des np array

    def fit_transform(self, X, y=None):
        """Apply both fit & transform"""
        self.fit(X)
        return self.transform(X)


class EmbeddingTransformer(BaseEstimator):
    """Constructs a transformer that apply an embedding mapping to Categorical columns"""

    def __init__(self, embedding, none_strategy='zeros'):
        '''Initialisation de la classe EmbeddingTransformer

        Args:
            embedding (str ou dict): embedding à utiliser
                - si dict -> ok, ready to go
                - si str -> chemin vers fichier à charger (json)
        Kwargs:
            none_strategy (str): strategy to fill elements not in embedding
                - zeros: only 0s
        Raises:
            TypeError: si embedding pas au bon format
            ValueError: si strategy "none" non reconnu
            ValueError: si l'embedding est de type str mais ne termine pas par .json
            FileNotFoundError: si le chemin vers l'embedding n'existe pas
        '''
        # Check format embedding
        if type(embedding) not in [str, dict]:
            raise TypeError("L'embedding doit être un dictionnaire ou un chemin de fichier JSON à charger")
        # Check none strategy
        allowed_strategies = ["zeros"]
        self.none_strategy = none_strategy
        if self.none_strategy not in allowed_strategies:
           raise ValueError("Can only use these strategies: {0} "
                            " got strategy={1}".format(allowed_strategies, self.none_strategy))

        # Si str, on load l'embedding
        if type(embedding) == str:
            if not embedding.endswith('.json'):
                raise ValueError(f"Le fichier {embedding} doit être un fichier .json")
            if not os.path.exists(embedding):
                raise FileNotFoundError(f"Le fichier {embedding} n'existe pas")
            with open(embedding, 'r', encoding='utf-8') as f:
                embedding = json.load(f)

        # Set embedding (format : {key : [embedding]})
        self.embedding = embedding
        # Get embedding size
        self.embedding_size = len(embedding[list(embedding.keys())[0]])
        # Other params
        self.n_features = None
        self.n_missed = 0

    def _validate_input(self, X):
        '''Function to validate input format

        Args:
            X: element to validate
        Returns:
            pd.DataFrame: X
        '''
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be a DataFrame or a numpy array")
        if self.n_features is not None and X.shape[1] != self.n_features:
            raise ValueError(f"Bad shape ({X.shape[1]} != {self.n_features})")

        # Copy obligatoire pour ne pas modifier l'original !
        if isinstance(X, pd.DataFrame):
            return X.copy(deep=True)
        else:
            return X.copy()

    def fit(self, X, y=None):
        """Fit transformer

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        Returns
        -------
        self
        """
        X = self._validate_input(X)
        # Si X np array, on transforme en dataframe
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.n_features = X.shape[1]

        # Nothing to do

        self.fitted_ = True
        return self

    def transform(self, X):
        """Transform X - embedding mapping
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        Raises
        ------
        ValueError: s'il manque des colonnes
        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        X = self._validate_input(X)
        # Si X np array, on transforme en dataframe
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        n_rows = X.shape[0]

        # Apply mapping
        new_df = pd.DataFrame()
        for col in X.columns:
            self.n_missed = 0 # On compte le nombre d'éléments non présents dans l'embedding
            tmp_serie = X[col].apply(self.apply_embedding) # Updates self.n_missed
            new_df = pd.concat([new_df, pd.DataFrame(tmp_serie.to_list())], axis=1)
            perc_missed = self.n_missed / n_rows * 100
            if perc_missed != 0:
                logger.warning(f"Attention, {self.n_missed} ({perc_missed} %) éléments non présents dans l'embedding pour la colonne {col}")

        return new_df.to_numpy() # Compatibilité -> on retourne des np array

    def fit_transform(self, X, y=None):
        """Apply both fit & transform"""
        self.fit(X)
        return self.transform(X)

    def apply_embedding(self, content):
        '''Apply embedding mapping

        Args:
            content: content on which apply embedding mapping
        Raises:
            ValueError: si la stratégie 'none' n'est pas reconnue
        Returns:
            list: applied embedding
        '''
        if content in self.embedding.keys():
            return self.embedding[content]
        else:
            self.n_missed += 1
            if self.none_strategy == 'zeros':
                return [0] * self.embedding_size
            else:
                raise ValueError(f"Stratégie {self.none_strategy} non reconnue")

    def get_feature_names(self, features_in, *args, **kwargs):
        """
        Return feature names for output features.
        output_feature_names : ndarray of shape (n_output_features,)
            Array of feature names.
        """
        check_is_fitted(self, 'fitted_')
        new_features = [f"emb_{feat}_{i}" for feat in features_in for i in range(self.embedding_size)]
        return np.array(new_features, dtype=object)


if __name__ == '__main__':
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")