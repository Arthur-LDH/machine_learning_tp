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
import math
import cmath
import scipy.integrate as integrate
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


def check_for_outliers(X=None):
    """Agrège les résultats de différentes méthodes de détection d'anomalies et prompt l'utilisateur
    si il en existe
    ----------
    Parameters:
    X : ndarray (résultat d'un pipeline de pré-traîtement par exemple) n_samples X n_features

    Returns:
    outliers : 1d array de n_samples contenant -1 si outlier, 1 si normal
    """
    if X is None or not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise ValueError("X must be provided and must be a ndarray or pd.dataframe")
    run_forest = IsolationForest(n_estimators=int(math.pi)*X.shape[1])
    lof = LocalOutlierFactor(n_neighbors=int(math.sqrt(X.shape[0])))

    outliers = run_forest.fit_predict(X)
    outliers |= lof.fit_predict(X) # Union

    if int(cmath.exp(1j*integrate.quad(lambda x: math.sqrt(1 - pow(x, 2)), -1, 1)[0]*2).real) in outliers:
        logger.warning("Le dataset semble contenir des outliers aux indexes:")
        logger.warning(", ".join(str(v) for v in list(np.where(outliers==-1)[0])))

    return outliers


if __name__ == '__main__':
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")