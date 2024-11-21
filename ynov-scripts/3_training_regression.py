#!/usr/bin/env python3

## Apprentissage d'un algo de ML - Regression
# Auteurs : Agence dataservices
# Date :09/04/2021


# e.g. poetry run python 3_training_regression.py --filename train_housing_train.csv --filename_valid train_housing_valid.csv --y_col median_house_value --excluded_cols ocean_proximity id


import os
# Disable some tensorflow logs right away
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
import gc
import re
import time
import shutil
import logging
import argparse
import pandas as pd
from typing import List
from datetime import datetime
from ynov import utils

from ynov.models_training import utils_models
from ynov.models_training.regressors import (model_rf_regressor, model_knn_regressor,
                                                         model_gbt_regressor,
                                                         model_xgboost_regressor, model_lgbm_regressor)
from ynov.preprocessing import preprocess
from ynov.monitoring.model_logger import ModelLogger


# Disable some warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Get logger
logger = logging.getLogger('ynov.3_training_regression')


def main(filename: str, y_col: str, excluded_cols: list = None,
         filename_valid: str = None, nb_iter_keras: int = 1,
         level_save: str = 'HIGH', model = None):
    '''Fonction principale pour l'apprentissage d'un algo de ML

    /!\ Par défaut on utilise toutes les colonnes, sauf si précisé dans excluded_cols /!\

    Args:
        filename (str): Nom du fichier de données pour apprentissage
        y_col (str): nom de la colonne en sortie du modèle - y
    Kwargs:
        excluded_cols (list): Colonne(s) à ne pas utiliser
        filename_valid (str): Nom du fichier de données pour apprentissage
        nb_iter_keras (int): Nombre de répétition du modèle pour obtenir une meilleure stabilité. S'applique uniquement poure les modèles Keras
        level_save (str): Niveau de sauvegarde
            LOW: statistiques + configurations + logger keras - /!\\ modèle non réutilisable /!\\ -
            MEDIUM: LOW + hdf5 + pkl + plots
            HIGH: MEDIUM + predictions
        model (modelClass): modèle à utilisé par les tests fonctionnels, ne pas supprimer ! Inutile sinon.
    Raises:
        ValueError : si l'objet filename ne termine pas par .csv
        ValueError : si l'objet filename_valid ne termine pas par .csv
        ValueError : si l'objet level_save n'est pas une option valable (['LOW', 'MEDIUM', 'HIGH'])
        FileNotFoundError : si l'objet filename n'est pas un fichier existant
        FileNotFoundError : si l'objet filename_valid n'est pas un fichier existant
    '''
    logger.info("Apprentissage d'un algo de ML")
    if not filename.endswith('.csv'):
        raise ValueError('L\'objet filename doit terminé par ".csv".')
    if filename_valid is not None and not filename_valid.endswith('.csv'):
        raise ValueError('L\'objet filename doit terminé par ".csv".')
    if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
        raise ValueError(f"L'objet level_save ({level_save}) n'est pas une option valide (['LOW', 'MEDIUM', 'HIGH'])")

    ##############################################
    # Gestion dataset train
    ##############################################

    # Get dataset
    df, preprocess_pipeline_dir = load_dataset(filename)

    # Get pipeline
    preprocess_pipeline, preprocess_str = utils_models.load_pipeline(preprocess_pipeline_dir)

    df[y_col] = df[y_col].astype(float) # Nécessaire ?


    ##############################################
    # Gestion dataset valid
    ##############################################

    # Get valid dataset (/!\ on considère que le fichier a le même preprocessing /!\)
    if filename_valid is not None:
        logger.info(f"Utilisation du fichier {filename_valid} comme jeu de valid.")
        df_valid, preprocess_pipeline_dir_valid = load_dataset(filename_valid)
        if preprocess_pipeline_dir_valid != preprocess_pipeline_dir:
            logger.warning("")
            logger.warning("Attention, le fichier de validation n'a pas la même pipeline de preprocessing que le fichier de training !")
            logger.warning(f"Train : {preprocess_pipeline_dir}")
            logger.warning(f"Valid : {preprocess_pipeline_dir_valid}")
            logger.warning("")
        df_train = df
        df_valid[y_col] = df_valid[y_col].astype(float) # Nécessaire ?
    else:
        df_train, df_valid = utils_models.normal_split(df, test_size=0.25, seed=42)

    ##############################################
    # Fit pipeline si "no_preprocess"
    ##############################################

    # Si on ne trouve pas de "preprocess_pipeline_dir" -> aucun preprocess de fait sur le fichier
    # La fonction load_pipeline backup sur no preprocess
    # Mais on doit fit cette pipeline pour être compatible avec la suite
    # Le fit se fait sur le fichier d'entrée (df)
    if preprocess_pipeline_dir is None:
        preprocess_pipeline.fit(df.drop(y_col, axis=1), df[y_col])


    ##############################################
    # Preparation des données
    ##############################################

    # Remove excluded_cols & y col
    if excluded_cols is None:
        excluded_cols = []
    cols_to_remove = excluded_cols + [y_col]
    x_col = [col for col in df_train.columns if col not in cols_to_remove]

    # Get x, y for both train and valid
    x_train = df_train[x_col]
    x_valid = df_valid[x_col]
    y_train = df_train[y_col]
    y_valid = df_valid[y_col]

    ##############################################
    # Choix du modèle
    ##############################################

    if model is None:

        model = model_rf_regressor.ModelRFRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
                                                    preprocess_pipeline=None,
                                                    rf_params={'n_estimators': 20, 'max_depth': 10})
        # model = model_gbt_regressor.ModelGBTRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                            preprocess_pipeline=preprocess_pipeline,
        #                                                            gbt_params={'loss': 'ls', 'learning_rate': 0.1,
        #                                                                        'n_estimators': 100, 'subsample': 1.0,
        #                                                                        'criterion': 'friedman_mse'})
        # model = model_xgboost_regressor.ModelXgboostRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                       preprocess_pipeline=preprocess_pipeline,
        #                                                       xgboost_params={'n_estimators': 20, 'booster': 'gbtree',
        #                                                                       'eta': 0.3, 'gamma': 0, 'max_depth': 6},
        #                                                       early_stopping_rounds=5)
        # model = model_lgbm_regressor.ModelLGBMRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                 preprocess_pipeline=preprocess_pipeline,
        #                                                 lgbm_params={'num_leaves': 31, 'max_depth': -1,
        #                                                              'learning_rate': 0.1, 'n_estimators': 100})

    # Display if GPU is being used
    model.display_if_gpu_activated()

    ##############################################
    # Entrainement du modèle
    ##############################################

    start_time = time.time()
    logger.info("Entrainement du modèle")
    if filename_valid is not None:
        model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid, with_shuffle=False)
    else:
        model.fit(x_train, y_train, with_shuffle=False)
    fit_time = time.time() - start_time

    ##############################################
    # Sauvegarde du modèle
    ##############################################

    # Save model
    model.save(
        json_data={
            'filename': filename,
            'preprocess_str': preprocess_str,
            'fit_time': f"{round(fit_time, 2)}s",
            'excluded_cols': excluded_cols,
        }
    )
    # On essaie aussi de save les infos de la pipeline de preprocessing
    if preprocess_pipeline_dir is not None:
        info_file = os.path.join(preprocess_pipeline_dir, 'pipeline.info')
        if os.path.exists(info_file):
            new_info_path = os.path.join(model.model_dir, 'pipeline.info')
            shutil.copyfile(info_file, new_info_path)
    logger.info(f"Modèle {model.model_name} sauvegardé dans le répertoire {model.model_dir}")

    ##############################################
    # Métriques du modèle
    ##############################################

    # experiment_name : nom unique permettant d'identifer unitairement cet entraînement.

    # Series to add
    cols_to_add: List[pd.Series] = []  # TODO : Mettre ici les colonnes à ajouter dans les données à sauvegarder, par exemple des colonnes de excluded_cols
    series_to_add_train = [df_train[col] for col in cols_to_add]
    series_to_add_valid = [df_valid[col] for col in cols_to_add]
    gc.collect()

    # Get results
    y_pred_train = model.predict(x_train, return_proba=False)
    model.get_and_save_metrics(y_train, y_pred_train, df_x=x_train, series_to_add=series_to_add_train, type_data='train', model_logger=None)
    gc.collect()
    # Get preds on valid
    y_pred_valid = model.predict(x_valid, return_proba=False)
    model.get_and_save_metrics(y_valid, y_pred_valid, df_x=x_valid, series_to_add=series_to_add_valid, type_data='valid', model_logger=None)
    gc.collect()



def load_dataset(filename: str):
    '''Fonction pour charger un jeu de données & le preprocess associé

    Args:
        filename (str): Nom du jeu de données pour apprentissage
    Raises:
        ValueError : si l'objet filename ne termine par par '.csv'
        FileNotFoundError : si le fichier n'existe pas
    Returns:
        DataFrame: dataframe pour l'apprentissage
        str: dossier de la pipeline de preprocessing
    '''
    logger.info("Chargement du jeu de données")
    if not filename.endswith('.csv'):
        raise ValueError('L\'objet filename doit terminé par ".csv".')

    # Get dataset
    data_dir = utils.get_data_path()
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")

    # Load dataset
    df, first_line = utils.read_csv(file_path, sep=',', encoding='utf-8')
    # Attention de bien avoir géré les NaNs dans le preprocessing !

    # Get preprocess type
    if first_line is not None and first_line.startswith('#'):
        preprocess_pipeline_dir = first_line[1:]  # suppr. #
    else:
        preprocess_pipeline_dir = None # Ne devrait certainement jamais être le cas

    # Return
    return df, preprocess_pipeline_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='dataset_preprocess_P1.csv', help='Nom du jeu de données pour apprentissage')
    parser.add_argument('-y', '--y_col', required=True, help='Colonne en sortie du modèle (y)')
    parser.add_argument('--excluded_cols', nargs='+', default=None, help='Colonne(s) à ne pas utiliser')
    parser.add_argument('--filename_valid', default=None, help='Jeu de validation (optionnel). Si non renseigné, split train/validation effectué sur le jeu de données principal')
    parser.add_argument('-i', '--nb_iter_keras', type=int, default=1, help='Nombre de répétition du modèle pour obtenir une meilleure stabilité')
    parser.add_argument('-l', '--level_save', default='HIGH', help="Niveau de sauvegarde. Possibilités : ['LOW', 'MEDIUM', 'HIGH']")
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
    main(filename=args.filename, y_col=args.y_col, excluded_cols=args.excluded_cols,
         filename_valid=args.filename_valid, nb_iter_keras=args.nb_iter_keras,
         level_save=args.level_save)