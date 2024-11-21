#!/usr/bin/env python3

## Utils - fonctions-outils
# Auteurs : Agence dataservices
# Date : 06/04/2021
#
# Fonctions :
# - read_csv -> Fonction pour lire un csv en analysant la première ligne
# - to_csv -> Fonction pour écrire un csv en gérant la première ligne
# - display_shape -> Affichage du nombre de lignes et nombre de colonnes d'une table
# - get_chunk_limits -> Fonction to get chunk limits from a pandas series or dataframe
# - trained_needed -> Décorateur pour s'assurer qu'un modèle à déjà été trained
# - get_configs_path -> Retourne le path du dossier de configs
# - get_data_path -> Retourne le path du dossier de data
# - get_models_path -> Retourne le path du dossier de models
# - get_pipelines_path -> Retourne le path du dossier des pipelines
# - get_ressources_path -> Retourne le path du dossier des ressources diverses
# - get_package_version -> Retourne la version courante du package
# - flatten -> Fonction pour applatir une liste d'éléments mixed (i.e. certains iterables, d'autres non)


import logging
import os
import json
import pkg_resources
import numpy as np
import pandas as pd
from collections.abc import Iterable


# Get logger
logger = logging.getLogger(__name__)

DIR_PATH = None  # IMPORTANT : VARIABLE A SET EN PROD POUR POINTER SUR LES REPERTOIRES DATA ET MODELS


# TODO: rajouter une fonction datalake_query pour récupérer des données du lac
# 11/06/2020
# Une fonction existait, mais:
#  - Problème de compatibilité pyodbc sur la plateforme HPE
#  - Plus fonctionnelle car mise à jour Kerberos


def read_csv(file_path: str, sep: str = ',', encoding: str = 'utf-8', **kwargs):
    '''Fonction pour lire un csv en analysant la première ligne

    Args:
        file_path (str): Chemin vers le fichier avec les données
    Kwargs:
        sep (str): Séparateur du fichier de données
        encoding (str): Encodage du fichier de données
        kwargs: kwargs pour pandas
    Raises:
        ValueError : si l'objet file_path ne termine pas par .csv
        FileNotFoundError : si l'objet file_path n'est pas un fichier existant
    Returns:
        pd.DataFrame: données
        str: première ligne du csv (None si commence pas par #), et sans sauts de ligne
    '''
    if not file_path.endswith('.csv'):
        raise ValueError('L\'objet file_path doit terminé par ".csv".')
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    # On récupère la première ligne
    with open(file_path, 'r', encoding=encoding) as f:
        first_line = f.readline()
    # On regarde si la première ligne comporte des métadatas
    has_metada = True if first_line.startswith('#') else False
    # On charge la dataset
    if has_metada:
        df = pd.read_csv(file_path, sep=sep, encoding=encoding, skiprows=1, **kwargs)
    else:
        df = pd.read_csv(file_path, sep=sep, encoding=encoding, **kwargs)

    # If no metadata, return only the dataframe
    if not has_metada:
        return df, None
    # Else process the first_line
    else:
        # Suppression saut de ligne
        if first_line is not None and first_line.endswith('\n'):
            first_line = first_line[:-1]
        # Suppression return carriage
        if first_line is not None and first_line.endswith('\r'):
            first_line = first_line[:-1]
        # Return
        return df, first_line


def to_csv(df: pd.DataFrame, file_path: str, first_line: str = None, sep: str = ',',
           encoding: str = 'utf-8', **kwargs):
    '''Fonction pour écrire un csv en gérant la première ligne

    Args:
        df (pd.DataFrame): données à écrire
        file_path (str): Chemin vers le fichier à créer
    Kwargs:
        first_line (str): Première ligne à écrire (sans saut de ligne, fait dans cette fonction)
        sep (str): Séparateur du fichier de données
        encoding (str): Encodage du fichier de données
        kwargs: kwargs pour pandas
    '''
    # On récupère la première ligne
    with open(file_path, 'w', encoding=encoding) as f:
        if first_line is not None:
            f.write(first_line + '\n')  # On ajoute la 1ère ligne si métadata
        df.to_csv(f, sep=sep, encoding=encoding, index=None, **kwargs)


def display_shape(df: pd.DataFrame):
    '''Affichage du nombre de lignes et nombre de colonnes d'une table

    Args:
        df (DataFrame): Table à analyser
    Raises:
        TypeError : si l'objet df n'est pas du type DataFrame
    '''
    logger.debug('Appel à la fonction utils.display_shape')
    # Display
    logger.info(f"Nombre de lignes : {df.shape[0]}. Nombre de colonnes : {df.shape[1]}.")


def get_chunk_limits(x, chunksize: int = 10000):
    '''Fonction to get chunk limits from a pandas series or dataframe

    Args:
        x (pd.Series or pd.DataFrame): Documents à traiter
    Kwargs:
        chunksize (int): taille des chunks
    Raises:
        TypeError: Si le document n'est pas dans une serie Pandas
        ValueError: Si chunksize positif
    Returns:
        list: chunk limits
    '''
    logger.debug('Appel à la fonction utils.get_chunk_limits')
    if type(x) not in [pd.Series, pd.DataFrame]:
        raise TypeError('L\'objet x doit être du type pd.Series ou pd.DataFrame.')
    if chunksize < 0:
        raise ValueError('L\'objet chunksize doit être positif.')
    # Processs
    if chunksize == 0 or chunksize >= x.shape[0]:
        chunks_limits = [(0, x.shape[0])]
    else:
        chunks_limits = [(i * chunksize, min((i + 1) * chunksize, x.shape[0]))
                         for i in range(1 + ((x.shape[0] - 1) // chunksize))]
    return chunks_limits


def trained_needed(function):
    '''Décorateur pour s'assurer qu'un modèle à déjà été trained

    Args:
        function (func): Fonction à décorer
    Returns:
        function: La fonction décorée
    '''
    logger.debug('Appel à la fonction utils.trained_needed')

    # Get wrapper
    def wrapper(self, *args, **kwargs):
        '''Wrapper'''
        if not self.trained:
            raise AttributeError(f"La fonction {function.__name__} ne peut pas être appelée tant que le model n'est pas fit")
        else:
            return function(self, *args, **kwargs)

    return wrapper


def get_configs_path():
    '''Retourne le path du dossier de configs

    Returns:
        str: path du dossier de configs
    '''
    logger.debug('Appel à la fonction utils.get_configs_path')
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def get_data_path():
    '''Retourne le path du dossier de data

    Returns:
        str: path du dossier de data
    '''
    logger.debug('Appel à la fonction utils.get_data_path')
    if DIR_PATH is None:
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'ynov-data')
    else:
        dir_path = os.path.join(os.path.abspath(DIR_PATH), 'ynov-data')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def get_models_path():
    '''Retourne le path du dossier de models

    Returns:
        str: path du dossier de models
    '''
    logger.debug('Appel à la fonction utils.get_models_path')
    if DIR_PATH is None:
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'ynov-models')
    else:
        dir_path = os.path.join(os.path.abspath(DIR_PATH), 'ynov-models')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def get_pipelines_path():
    '''Retourne le path du dossier des pipelines

    Returns:
        str: path du dossier de pipelines
    '''
    logger.debug('Appel à la fonction utils.get_pipelines_path')
    if DIR_PATH is None:
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'ynov-pipelines')
    else:
        dir_path = os.path.join(os.path.abspath(DIR_PATH), 'ynov-pipelines')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def get_ressources_path():
    '''Retourne le path du dossier de models

    Returns:
        str: path du dossier de models
    '''
    logger.debug('Appel à la fonction utils.get_ressources_path')
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'ynov-ressources')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def get_package_version():
    '''Retourne la version courante du package

    Returns:
        str: version du package
    '''
    version = pkg_resources.get_distribution('ynov').version
    return version


def flatten(my_list: Iterable):
    '''Fonction pour applatir une liste d'éléments mixed (i.e. certains iterables, d'autres non)

    e.g. [[1, 2], 3, [4]] -> [1, 2, 3, 4]

    From : https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists

    Args:
        my_list (list): liste à traiter
    Results:
        generator: liste flattened (format generator)
    '''
    for el in my_list:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


# Encodeur JSON permettant de gérer les objets numpy
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")