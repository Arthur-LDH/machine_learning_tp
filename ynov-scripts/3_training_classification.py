#!/usr/bin/env python3

## Apprentissage d'un algo de ML - Classification
# Auteurs : Agence dataservices
# Date :09/04/2021


# e.g. poetry run python 3_training_classification.py --filename dataset_train_preprocess_P1.csv --filename_valid dataset_valid_preprocess_P1.csv --y_col Survived


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
from ynov.models_training.classifiers import (model_rf_classifier, model_dense_classifier,
                                                          model_ridge_classifier, model_logistic_regression_classifier,
                                                          model_sgd_classifier, model_svm_classifier, model_knn_classifier,
                                                          model_gbt_classifier, model_lgbm_classifier, model_xgboost_classifier)
from ynov.preprocessing import preprocess
from ynov.monitoring.model_logger import ModelLogger


# Disable some warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Get logger
logger = logging.getLogger('ynov.3_training_classification')


def main(filename: str, y_col: list, excluded_cols: list = None,
         filename_valid: str = None, min_rows: int = None, nb_iter_keras: int = 1,
         level_save: str = 'HIGH', model = None):
    '''Fonction principale pour l'apprentissage d'un algo de ML

    /!\ Par défaut on utilise toutes les colonnes, sauf si précisé dans excluded_cols /!\

    Args:
        filename (str): Nom du fichier de données pour apprentissage
        y_col (list): nom des colonnes à utiliser pour l'apprentissage - y
    Kwargs:
        excluded_cols (list): Colonne(s) à ne pas utiliser
        min_rows (int): Nombre minimums de données dans le jeu de données par classe
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

    ### TODO
    ### TOUS les modèles attendent en cible des données :
    ###
    ###   - OHE (integers) si classfication multilabel
    ###     e.g.
    ###         col1 col2 col3
    ###            1    0   1
    ###            0    0   0
    ###            0    1   0
    ###
    ###   - string si classification monolabel (même si 0/1 -> '0'/'1')
    ###     e.g.
    ###         target
    ###           toto
    ###           titi
    ###           tata
    ###           toto
    ###
    ###
    ### Ci dessous, quelques exemples de preprocessing possibles en fonction du jeu de données en entrées :
    ###
    ### - multilabel -> Grouper des données :
    ### ********
    ### y_col = y_col[0]
    ### # Selection des colonnes d'information
    ### info_cols = [col for col in list(df.columns) if col != y_col]
    ### # On groupe par ces colonnes, et on applique une transformation 'tuple' en ignorant les strings vides
    ### tuple_transformation = lambda x: tuple([_ for _ in x if _ != ''])
    ### df = pd.DataFrame(df.groupby(info_cols)[y_col].apply(tuple_transformation)).reset_index()
    ### ********
    ###
    ### - multilabel -> Transformer tuples en OHE
    ### ********
    ### y_col = y_col[0]
    ### # Make sure column is tuples and not strings
    ### from ast import literal_eval
    ### df[y_col] = df[y_col].apply(lambda x: literal_eval(x))
    ### # Transform to OHE
    ### df, y_col = utils_models.preprocess_model_multilabel(df, y_col, classes=None)
    ### ********
    ###
    ### TODO

    # Check if multi-label
    if len(y_col) > 1:
        multi_label = True
        # TODO : 07/09/2020 -> package non compatible multi-class/multi-label
        try:
            df[y_col] = df[y_col].astype(int)  # Need to cast OHE encoded var into integers
            for col in y_col:
                assert sorted(df[col].unique()) == [0, 1]
        except:
            logger.error("Impossible de cast les colonnes en entiers (0/1)")
            logger.error("Avez-vous bien réaliser la transformation en OHE ?")
            logger.error("Au 07/09/2020, le package n'est pas compatible multi-class/multi-label")
            logger.error("Nous vous conseillons de transformer les colonnes multi-class en OHE, ou d'entrainer plusieurs modèles multi-class/mono-label")
    else:
        multi_label = False
        y_col = y_col[0]
        df[y_col] = df[y_col].astype(str)

    # Remove small classes if wanted (only possible if not multilabel)
    if min_rows is not None and not multi_label:
        df = utils_models.remove_small_classes(df, y_col, min_rows=min_rows)

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
        ### TODO
        ### Gérer df_valid comme pour df_train
        ### TODO
        df_train = df
        # Manage OHE format (ensure int)
        if multi_label:
            df_valid[y_col] = df_valid[y_col].astype(int)
        else:
            df_valid[y_col] = df_valid[y_col].astype(str)
    else:
        if not multi_label:
            df_train, df_valid = utils_models.stratified_split(df, y_col, test_size=0.25, seed=42)
            # df_train, df_valid = utils_models.hierarchical_split(df, y_col, test_size=0.25, seed=42)
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

    # Remove excluded_cols & y cols
    if excluded_cols is None:
        excluded_cols = []
    if type(y_col) == list:
        cols_to_remove = excluded_cols + y_col
    else:
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

    # TODO
    # Si vous souhaitez continuer l'entrainement d'un modèle existant :
    # model, _ = utils_models.load_model("dir_model")
    # Libre à vous de modifier certains paramètres comme la batch size, le learning rate (via _get_learning_rate_scheduler de préférence), etc.
    # Attention d'utiliser le même preprocessing pour les données en entrée d'apprentissage
    # TODO

    # TODO:
    # Si vous voulez effectuer une recherche d'hyperparmètres :
    # model_cls = model_tfidf_svm.ModelTfidfSvm # La classe de modèle à utiliser
    # model_params = {'x_col': x_col, 'y_col': y_col, 'multi_label': multi_label} # Paramètres "fixes" du modèle
    # hp_params = {'tfidf_params': [{'analyzer': 'word', 'ngram_range': (1, 2), "max_df":0.1}, {'analyzer': 'word', 'ngram_range': (1, 3), "max_df":0.1}]} # L'ensemble des paramètres à tester
    # scoring_fn = "f1" # La fonction d'évaluation pour récupérer le "best model", à maximiser
    # kwargs_fit = {'x_train':x_train, 'y_train': y_train, 'with_shuffle': False} # Inutile de mettre x_valid & y_valid car crossvalidation sur le train
    # n_splits = 5 # Nombre de crossvalidation
    # model = utils_models.search_hp_cv_classifier(model_cls, model_params, hp_params, scoring_fn, kwargs_fit, n_splits=n_splits) # Retourne un model avec les meilleurs params (to be fitted on the whole dataset)
    # TODO:

    if model is None:
        model = model_ridge_classifier.ModelRidgeClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
                                                            preprocess_pipeline=preprocess_pipeline,
                                                            ridge_params={'alpha': 1.0},
                                                            multi_label=multi_label)
        # model = model_logistic_regression_classifier.ModelLogisticRegressionClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                                                preprocess_pipeline=preprocess_pipeline,
        #                                                                                lr_params={'penalty': 'l2', 'C': 1.0, 'max_iter': 100},
        #                                                                                multi_label=multi_label)
        # model = model_svm_classifier.ModelSVMClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                 preprocess_pipeline=preprocess_pipeline,
        #                                                 svm_params={'C': 1.0, 'kernel': 'linear'},
        #                                                 multi_label=multi_label)
        # model = model_sgd_classifier.ModelSGDClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                 preprocess_pipeline=preprocess_pipeline,
        #                                                 sgd_params={'loss': 'hinge', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
        #                                                 multi_label=multi_label)
        # model = model_knn_classifier.ModelKNNClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                 preprocess_pipeline=preprocess_pipeline,
        #                                                 knn_params={'n_neighbors': 7, 'weights': 'distance'},
        #                                                 multi_label=multi_label)
        # model = model_rf_classifier.ModelRFClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                               preprocess_pipeline=preprocess_pipeline,
        #                                               rf_params={'n_estimators': 50, 'max_depth': 5},
        #                                               multi_label=multi_label)
        # model = model_gbt_classifier.ModelGBTClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                 preprocess_pipeline=preprocess_pipeline,
        #                                                 gbt_params={'loss': 'deviance', 'learning_rate': 0.1,
        #                                                             'n_estimators': 100, 'subsample': 1.0,
        #                                                             'criterion': 'friedman_mse'},
        #                                                 multi_label=multi_label)
        # model = model_xgboost_classifier.ModelXgboostClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                         preprocess_pipeline=preprocess_pipeline,
        #                                                         xgboost_params={'n_estimators': 20, 'booster': 'gbtree',
        #                                                                         'eta': 0.3, 'gamma': 0, 'max_depth': 6},
        #                                                         early_stopping_rounds=5,
        #                                                         multi_label=multi_label)
        # model = model_lgbm_classifier.ModelLGBMClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                   preprocess_pipeline=preprocess_pipeline,
        #                                                   lgbm_params={'num_leaves': 31, 'max_depth': -1,
        #                                                                'learning_rate': 0.1, 'n_estimators': 100},
        #                                                   multi_label=multi_label)
        # model = model_dense_classifier.ModelDenseClassifier(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                     preprocess_pipeline=preprocess_pipeline,
        #                                                     batch_size=64, epochs=99, patience=5,
        #                                                     multi_label=multi_label, nb_iter_keras=nb_iter_keras)


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
            'min_rows': min_rows,
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

    # Création d'un logger permettant d'enregistrer les métrics du modèle sur mlflow
    # tracking_uri : se rapprocher d'un ops / team socle pour obtenir l'url mlflow sur la plateforme IA
    # experiment_name : nom unique permettant d'identifer unitairement cet entraînement.
    # -> par défaut dossier du modèle, mais attention si vous definissez un model_dir custom (risque de ne plus avoir un id unique)
    model_logger = ModelLogger(
        tracking_uri="http://mlflow01-poc-pe01.datasvc01.k8s.pole-emploi.intra",  # l'URI peut changer en fonction des évolutions de la plateforme
        experiment_name=f"ynov",
    )
    model_logger.set_tag('model_name', f"{os.path.basename(model.model_dir)}")
    # model_logget.set_tag(key, value) pour enregistrer des informations complémentaires exogènes au modèle
    # par exemple l'embedding utilisé ou les données utilisées
    # model_logget.log_param(key, value) pour enregistrer des informations relatives aux paramètres du modèle
    # par exemple le learning rate

    # Series to add
    cols_to_add: List[pd.Series] = []  # TODO : Mettre ici les colonnes à ajouter dans les données à sauvegarder, par exemple des colonnes de excluded_cols
    series_to_add_train = [df_train[col] for col in cols_to_add]
    series_to_add_valid = [df_valid[col] for col in cols_to_add]
    gc.collect()

    # Get results
    y_pred_train = model.predict(x_train, return_proba=False)
    #model_logger.set_tag(key='type_metric', value='train')
    model.get_and_save_metrics(y_train, y_pred_train, df_x=x_train, series_to_add=series_to_add_train, type_data='train', model_logger=model_logger)
    gc.collect()
    # Get preds on valid
    y_pred_valid = model.predict(x_valid, return_proba=False)
    #model_logger.set_tag(key='type_metric', value='valid')
    model.get_and_save_metrics(y_valid, y_pred_valid, df_x=x_valid, series_to_add=series_to_add_valid, type_data='valid', model_logger=model_logger)
    gc.collect()

    # Stop MLflow
    model_logger.stop_run()


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
    parser.add_argument('-y', '--y_col', nargs='+', required=True, help='Colonne(s) en sortie du modèle (y)')
    parser.add_argument('--excluded_cols', nargs='+', default=None, help='Colonne(s) à ne pas utiliser')
    parser.add_argument('-m', '--min_rows', type=int, default=None, help='Nombre minimums de données dans le jeu de données par classe.')
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
         min_rows=args.min_rows, filename_valid=args.filename_valid,
         nb_iter_keras=args.nb_iter_keras, level_save=args.level_save)