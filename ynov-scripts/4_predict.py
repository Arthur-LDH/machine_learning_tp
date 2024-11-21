#!/usr/bin/env python3

# Application d'un algo de ML pour obtenir des prédictions
# Auteurs : Agence dataservices
# Date : 14/04/2021


# e.g. poetry run python 4_predict.py --filename test_housing.csv --model model_rf_regressor_2024_11_16-18_40_57


import os
import uuid
import json
import dill as pickle
import logging
import argparse
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from datetime import datetime

from ynov import utils
from ynov.preprocessing import preprocess
from ynov.models_training import model_class, utils_models


# Get logger
logger = logging.getLogger('ynov.4_predict')


def main(filename: str, sep: str, encoding: str, model_dir: str, y_col: list = None):
    '''Fonction principale pour l'application d'un algo de ML pour obtenir des prédictions

    Args:
        filename (str): Nom du fichier de données pour le test
        sep (str): séparateur du fichier de données
        encoding (str): Encodage du fichier de données
        model_dir (str): Nom du modèle à utiliser
    Kwargs:
        y_col (list): Colonne(s) du dataframe à utiliser pour y_true (def: None)
    Raises:
        ValueError : si l'objet filename ne termine pas par .csv
        FileNotFoundError : si l'objet filename n'est pas un fichier existant
    '''
    if not filename.endswith('.csv'):
        raise ValueError('L\'objet filename doit terminé par ".csv".')

    # Process
    data_dir = utils.get_data_path()
    df_path = os.path.join(data_dir, filename)
    if not os.path.isfile(df_path):
        raise FileNotFoundError(f"Le fichier {filename} n'existe pas.")

    # Load model
    logger.info("Chargement du modèle")
    model, model_conf = utils_models.load_model(model_dir=model_dir)

    # Load dataset & preprocess it
    logger.info("Chargement & preprocessing du dataset")
    df, df_prep = load_dataset_test(df_path=df_path, sep=sep, encoding=encoding, model=model)

    # Try to keep only needed/wanted columns
    # It is useful if --excluded_cols used in training
    if all([col in df_prep.columns for col in model.x_col]):
        df_prep = df_prep[model.x_col]

    # Get predictions
    logger.info("Prédictions sur le jeu de données")
    y_pred = list(model.predict(df_prep, return_proba=False))
    # Get "unique" preds col
    predictions_col = 'predictions' if 'predictions' not in df.columns else f'predictions_{str(uuid.uuid4())[:8]}'
    # Add preds to original - non preprocessed - dataframe
    # TODO : on est certain que c'est la version non preprocessed qui doit être saved ?
    df[predictions_col] = list(model.inverse_transform(np.array(y_pred)))

    # Save result
    logger.info("Sauvegarde")
    save_dir = os.path.join(data_dir, 'predictions', Path(filename).stem, datetime.now().strftime("predictions_%Y_%m_%d-%H_%M_%S"))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_file = "predictions.csv"
    file_path = os.path.join(save_dir, save_file)
    df = df[["id", "predictions"]]
    df.to_csv(file_path, sep=',', encoding='utf-8', index=None)

    # Also save some info into a configs file
    conf_file = 'configurations.json'
    conf_path = os.path.join(save_dir, conf_file)
    conf = {
        'model_dir': model_dir,
        'preprocess_str': model_conf['preprocess_str'],
        'model_name': model_conf['model_name']
    }
    with open(conf_path, 'w', encoding='utf-8') as f:
        json.dump(conf, f, indent=4)

    # Get metrics if y_col is not None
    if y_col is not None:

        ### TODO
        ### Faire en sorte d'avoir le bon format en entrée (comme dans 2_training.py)
        ### TODO

        if model.model_type == 'classifier':
            if len(y_col) > 1:
                y_true = df[y_col].astype(int)  # Need to cast OHE encoded var into integers
            else:
                y_true = df[y_col[0]].astype(str)
        else:
            if len(y_col) > 1:
                raise NotImplementedError("Les modèles de type regression ne supporte pas (encore) le multioutput")
            else:
                y_true = df[y_col[0]].astype(float)

        cols_to_add: List[pd.Series] = []  # TODO : Mettre ici les colonnes à ajouter dans les données à sauvegarder
        series_to_add = [df[col] for col in cols_to_add]
        # Change model directory to save dir & get preds
        model.model_dir = save_dir
        model.get_and_save_metrics(y_true, y_pred, series_to_add=series_to_add, type_data='with_y_true')


def load_dataset_test(df_path: str, sep: str, encoding: str, model):
    ''' Fonction pour charger le dataset de test

    Args:
        df_path (str): Chemin du fichier .csv
        sep (str): séparateur du fichier de données
        encoding (str): Encodage du fichier de données
        model (ModelClass): modèle à utiliser pour les prédictions
    Raises:
        ValueError : si l'objet df_path ne termine pas par .csv
        FileNotFoundError : si le chemin df_path ne pointe pas sur fichier existant
    Returns:
        pd.DataFrame: dataframe chargée
        pd.DataFrame: dataframe chargée - preprocessed
    '''
    if not os.path.isfile(df_path):
        raise FileNotFoundError(f"Le fichier {df_path} n'existe pas.")

    # Get dataset
    df, _ = utils.read_csv(df_path, sep=sep, encoding=encoding)

    # Apply preprocessing
    if model.preprocess_pipeline is not None:
        df_prep = utils_models.apply_pipeline(df, model.preprocess_pipeline)
    else:
        df_prep = df.copy()
        logger.warning("On ne trouve pas de pipeline de preprocessing - on considère no preprocessing, mais ce n'est pas normal !")

    # Return
    return df, df_prep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='newdata.csv', help='Nom du jeu de données pour les prédictions.')
    parser.add_argument('--sep', default=',', help='Séparateur utilisé dans le jeu de données.')
    parser.add_argument('--encoding', default="utf-8", help='Encoding du csv')
    parser.add_argument('-y', '--y_col', nargs='+', default=None, help='Colonne(s) en sortie du modèle (y)')
    # model_X should be the model's directory name: e.g. model_tfidf_svm_2019_12_05-12_57_18
    parser.add_argument('-m', '--model_dir', default=None, help='Nom du model à utiliser')
    parser.add_argument('--force_cpu', dest='on_cpu', action='store_true', help="Entrainement forcé sur CPU (= pas GPU)")
    parser.set_defaults(on_cpu=False)
    args = parser.parse_args()
    # On check si on ne force pas le CPU
    if args.on_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        logger.info("----------------------------------------")
        logger.info("UTILISATION CPU FORCEE PAR L'UTILISATEUR")
        logger.info("----------------------------------------")
    # Main
    main(filename=args.filename, sep=args.sep, encoding=args.encoding, model_dir=args.model_dir, y_col=args.y_col)