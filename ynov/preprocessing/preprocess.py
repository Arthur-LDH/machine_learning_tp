#!/usr/bin/env python3

## Fonctions de preprocessing
# Auteurs : Agence dataservices
# Date : 16/12/2019


import re
import logging
import functools
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, KBinsDiscretizer, Binarizer, PolynomialFeatures, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, _VectorizerMixin
from sklearn.feature_selection import SelectKBest, SelectorMixin
from sklearn.impute import SimpleImputer
from ynov.preprocessing import column_preprocessors


# Get logger
logger = logging.getLogger(__name__)


def get_pipelines_dict():
    '''Fonction pour récupérer un dictionnaire des preprocessing possibles

    Returns:
        dict: dictionnaire des preprocessing
    '''
    pipelines_dict = {
        'no_preprocess': ColumnTransformer([('identity', FunctionTransformer(lambda x: x), make_column_selector())]),  # - /!\ NE PAS SUPPRIMER -> nécessaire pour compatibilité /!\ -
        'preprocess_P1': preprocess_P1(),  # Exemple donné avec le template
        # 'preprocess_AUTO' : preprocess_auto(), # Preprocessing automatisé à part de statistiques sur les données
        #  'preprocess_P2': preprocess_P2 , ETC ...
    }
    return pipelines_dict


def get_pipeline(preprocess_str: str):
    '''Fonction pour renvoyer un pipeline à utiliser

    Args:
        preprocess_str (str): type de preprocess
    Raises:
        ValueError: Si le type de preprocess n'est pas valide
    Returns:
        ColumnTransfomer: pipeline à utiliser pour le preprocessing
    '''
    # Process
    pipelines_dict = get_pipelines_dict()
    if preprocess_str not in pipelines_dict.keys():
        raise ValueError(f"Le preprocess {preprocess_str} n'est pas connu.")
    # Get pipeline
    pipeline = pipelines_dict[preprocess_str]
    # Return
    return pipeline


def preprocess_P1():
    '''Fonction principale pour preprocess le jeu de données

    Returns:
        pd.DataFrame: DataFrame modifiée (features uniquement)
    '''
    numeric_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    text_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=5))

    # Check https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html
    # and https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes
    # to understand make_column_selector

     # /!\ ICI EXEMPLE /!\
     # Bonne pratique : utiliser directement les noms de colonnes plutôt qu'un "sélecteur"
     # WARNING: la pipeline textuelle est prévue sur une colonne 'text' -> A adapter à votre projet

    # Par défault, on laisse juste le preprocess des colonnes numériques
    transformers = [
        ('num', numeric_pipeline, make_column_selector(dtype_include='number')),
        # ('cat', cat_pipeline, make_column_selector(dtype_include='category')), # Pour convertir une colonne en 'category' -> df["A"].astype("category")
        # ('text', text_pipeline, 'text'), # CountVectorizer possible sur une seule colonne à la fois
    ]

    pipeline = ColumnTransformer(transformers, remainder='drop') # Use remainder='passthrough' to keep all other columns (déconseillé)

    return pipeline


# TODO
def preprocess_auto():
    '''Fonction principale pour preprocess le jeu de données
    automatiquement. Différentes fonctions sont appliquées en fonction de stats sur les données

    Returns:
        pd.DataFrame: DataFrame modifiée (features uniquement)
    '''
    # Numeric :
    # 1) SimpleImputer()
    # 2) Si abs(skew) > 2 && pctl(90) - pctl(10) > 10^3 => logtransform
    # 3) StandardScaler()
    # Categorical :
    # 1) SimpleImputer()
    # 2) Si #cat > 5; on accumule les intances les moins représentées dans une meta catégorie "other"
    # 3) OneHot
    pass


def retrieve_columns_from_pipeline(df: pd.DataFrame, pipeline: ColumnTransformer):
    '''Function to retrieve columns name after preprocessing

    Args:
        df (pd.DataFrame): dataframe après preprocessing (sans target)
        pipeline (ColumnTransformer): pipeline utilisée
    Returns:
        pd.DataFrame: dataframe avec colonnes
    '''
    #EXPERIMENTAL : on try catch tout ça
    try:
        # Check if fitted:
        if not hasattr(pipeline, '_columns'):
            raise AttributeError("La pipeline doit être fit pour utiliser la fonction retrieve_columns_from_pipeline")
        new_columns = get_ct_feature_names(pipeline)
        assert len(new_columns) == df.shape[1], "On ne retrouve pas le même nombre de colonnes" +\
                                                f" entre la DataFrame preprocessed ({df.shape[1]})" +\
                                                f" et la pipeline ({len(new_columns)})."
        df.columns = new_columns
    except Exception as e:
        logger.error("On n'annule la récupération des noms de colonnes (expérimental)")
        logger.error("On continue quand même")
        logger.error(repr(e))
    return df


def get_feature_out(estimator, features_in):
    '''Fonction pour récupérer le nom d'une colonne en sortie d'un estimator
    From : https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api
    '''
    if hasattr(estimator, 'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}' \
                for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(features_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(features_in)[estimator.get_support()]
    else:
        return features_in


def get_ct_feature_names(ct):
    '''Fonction pour récupérer les noms des colonnes en sortie d'un ColumnTransfomer
    From : https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api
    '''
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != 'remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    if type(step) == tuple:
                        step = step[1]
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator == 'passthrough':
            output_features.extend(ct._feature_names_in[features])

    return output_features


if __name__ == '__main__':
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")