#!/usr/bin/env python3

## Séparation d'un jeu de données en train/valid/test
# Auteurs : Agence dataservices
# Date : 13/05/2020
#
# Ex: poetry run python 0_split_train_valid_test.py -f train_housing.csv --perc_train 70 --perc_valid 30 --perc_test 0

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from ynov import utils
from ynov.models_training import utils_models


# Get logger
logger = logging.getLogger('ynov.0_split_train_valid_test')


def main(filename: str, split_type: str, perc_train: float, perc_valid: float, perc_test: float,
         y_col=None, sep: str = ',', encoding: str = 'utf-8',
         seed: int = None):
    '''Fonction principale pour extraire un subset de données depuis un fichier

    Args:
        filename (str): Nom du fichier de données à traiter
        split_type (str): Type de split à réaliser (random, stratified)
        perc_train (float): Fraction jeu de Train
        perc_valid (float): Fraction jeu de Validation
        perc_test (float): Fraction jeu de Test
    Kwargs:
        y_col (str ou int): Colonne à utiliser pour split stratified
        sep (str): Séparateur du fichier de données
        encoding (str): Encodage du fichier de données
        seed (int): seed à utiliser pour reproduire les résultats
    Raises:
        TypeError : si l'objet y_col n'est pas du type str ou int
        ValueError : si l'objet filename ne termine pas par .csv
        FileNotFoundError : si l'objet filename n'est pas un fichier existant
        ValueError : si l'objet split_type n'est pas égal à 'random' ou 'stratified'
        ValueError : si l'objet split_type est égal à 'stratified' mais que y_col n'est pas set
    '''
    logger.info(f"Split train/valid/test du fichier {filename}")
    if y_col is not None and type(y_col) not in [str, int]:
        raise TypeError('L\'objet y_col doit être du type str ou int')
    if not filename.endswith('.csv'):
        raise ValueError('L\'objet filename doit terminé par ".csv".')
    if split_type not in ['random', 'stratified']:
        raise ValueError("L'objet split_type doit être égal à 'random' ou 'stratified'")
    if split_type == 'stratified' and y_col is None:
        raise ValueError("y_col doit être set avec l'option 'stratified'")

    # Set seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Get path
    data_dir = utils.get_data_path()
    file_path = os.path.join(data_dir, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    # Get dataframe
    # TODO: à vérifier -> on load tout en string pour éviter les erreurs + fillna
    df, first_line = utils.read_csv(file_path, sep=sep, encoding=encoding, dtype=str)
    df = df.fillna('')

    # Normalisation perc_train, perc_valid, perc_test
    perc_sum = perc_train + perc_valid + perc_test
    perc_train = perc_train / perc_sum
    perc_valid = perc_valid / perc_sum
    perc_test = perc_test / perc_sum
    logger.info(f'Pourcentage train : {perc_train * 100}%')
    logger.info(f'Pourcentage validation : {perc_valid * 100}%')
    logger.info(f'Pourcentage test : {perc_test * 100}%')

    # Split
    if split_type == 'random':
        df_train, df_valid, df_test = split_random(df, perc_train, perc_valid, perc_test)
    else: # split_type == 'stratified'
        df_train, df_valid, df_test = split_stratified(df, y_col, perc_train, perc_valid, perc_test, seed=seed)

    # Display info
    logger.info(f"Nombre de ligne dans le dataset d'origine : {df.shape[0]}")
    logger.info(f"Nombre de ligne dans le dataset de train : {df_train.shape[0]} ({df_train.shape[0] / df.shape[0] * 100} %)")
    logger.info(f"Nombre de ligne dans le dataset de validations : {df_valid.shape[0]} ({df_valid.shape[0] / df.shape[0] * 100} %)")
    logger.info(f"Nombre de ligne dans le dataset de test : {df_test.shape[0]} ({df_test.shape[0] / df.shape[0] * 100} %)")

    # Save
    basename = Path(filename).stem
    utils.to_csv(df_train, os.path.join(data_dir, f"{basename}_train.csv"), first_line=first_line,
                 sep=',', encoding='utf-8')
    utils.to_csv(df_valid, os.path.join(data_dir, f"{basename}_valid.csv"), first_line=first_line,
                 sep=',', encoding='utf-8')
    utils.to_csv(df_test, os.path.join(data_dir, f"{basename}_test.csv"), first_line=first_line,
                 sep=',', encoding='utf-8')


def split_random(df: pd.DataFrame, perc_train: float, perc_valid: float, perc_test: float):
    '''Fonction pour faire un split random

    Args:
        df (pd.DataFrame): données à écrire
        perc_train (float): Fraction jeu de Train
        perc_valid (float): Fraction jeu de Validation
        perc_test (float): Fraction jeu de Test
    '''
    # On s'assure de ne pas modifier la dataframe d'origine (utile ?)
    df = df.copy().reset_index(drop=True)
    # Sélection train
    df_train = df.sample(frac=perc_train)
    idx_train = df_train.index
    df = df[~df.index.isin(idx_train)]
    # Sélection valid (avec update perc_valid)
    perc_valid = perc_valid / (perc_test + perc_valid)
    df_valid = df.sample(frac=perc_valid)
    idx_valid = df_valid.index
    df = df[~df.index.isin(idx_valid)]
    # Sélection test (le reste)
    df_test = df
    # Return
    return df_train, df_valid, df_test


def split_stratified(df: pd.DataFrame, y_col, perc_train: float, perc_valid: float, perc_test: float, seed: int = None):
    '''Fonction pour faire un split stratified

    Args:
        df (pd.DataFrame): données à écrire
        y_col (str ou int): Colonne à utiliser pour split stratified
        perc_train (float): Fraction jeu de Train
        perc_valid (float): Fraction jeu de Validation
        perc_test (float): Fraction jeu de Test
    Kwargs:
        seed (int): seed à utiliser pour reproduire les résultats
    Raises:
        TypeError : si l'objet y_col n'est pas du type str ou int
    '''
    if type(y_col) not in [str, int]:
        raise TypeError('L\'objet y_col doit être du type str ou int')
    # On s'assure de ne pas modifier la dataframe d'origine (utile ?)
    df = df.copy().reset_index(drop=True)
    # Get random seed
    if seed is None:
        seed = random.randint(1, 100000)
    # Sélection train
    df_train, df_v_plus_t = utils_models.stratified_split(df, y_col, test_size=perc_valid + perc_test, seed=seed)
    # Update perc_test
    perc_test = perc_test / (perc_test + perc_valid)
    df_valid, df_test = utils_models.stratified_split(df_v_plus_t, y_col, test_size=perc_test, seed=seed)
    # On rajoute les index "perdu" (min_classes)
    df_lost = df[~df.index.isin(df_train.index.append(df_test.index).append(df_valid.index))]
    df_train = pd.concat([df_train, df_lost], sort=False)
    # Return
    return df_train, df_valid, df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='dataset.csv', help='Nom du jeu de données à traiter.')
    parser.add_argument('--split_type', default='random', help='Type de split à effectuer. Possibilités: random, stratified')
    parser.add_argument('--perc_train', default=0.6, type=float, help='Repartition du jeu de train')
    parser.add_argument('--perc_valid', default=0.2, type=float, help='Repartition du jeu de valid')
    parser.add_argument('--perc_test', default=0.2, type=float, help='Repartition du jeu de test')
    parser.add_argument('--y_col', default=None, help='Colonne à utiliser pour split stratified')
    parser.add_argument('--sep', default=',', help='Séparateur utilisé dans le jeu de données.')
    parser.add_argument('--encoding', default="utf-8", help='Encoding du csv')
    parser.add_argument('--seed', default=None, type=int, help="Seed à utiliser pour reproduire les résultats. Defaut: None")
    args = parser.parse_args()
    main(filename=args.filename, split_type=args.split_type, perc_train=args.perc_train,
         perc_valid=args.perc_valid, perc_test=args.perc_test, y_col=args.y_col,
         sep=args.sep, encoding=args.encoding, seed=args.seed)